import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #FF8C8C;
    }
    .description {
        font-size: 1.2rem;
    }
    .feature-header {
        font-size: 1.5rem;
        color: #FF6B6B;
        font-weight: bold;
    }
    .feature-impact {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    .prediction-result {
        font-size: 1.8rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .high-risk {
        background-color: #FFCCCB;
        color: #CC0000;
    }
    .low-risk {
        background-color: #CCFFCC;
        color: #006600;
    }
    .footnote {
        font-size: 0.8rem;
        color: #888888;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_resources():
    """Load the trained model and related resources"""
    nn_model = load_model('models/heart_model.keras')
    xgb_model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    explanation_data = joblib.load('models/explanation_data.pkl')
    ensemble_meta = joblib.load('models/ensemble_meta.pkl')
    
    # Create ensemble model wrapper
    class EnsembleModel:
        def __init__(self, nn_model, xgb_model, blend_weights):
            self.nn_model = nn_model
            self.xgb_model = xgb_model
            self.blend_weights = blend_weights
        
        def predict(self, X):
            # Neural Network predictions
            nn_preds = self.nn_model.predict(X).flatten()
            
            # XGBoost predictions
            xgb_preds = self.xgb_model.predict_proba(X)[:, 1]
            
            # Blend predictions
            ensemble_preds = self.blend_weights[0] * nn_preds + self.blend_weights[1] * xgb_preds
            
            return ensemble_preds.reshape(-1, 1)
    
    # Create the ensemble model
    ensemble_model = EnsembleModel(nn_model, xgb_model, ensemble_meta['blend_weights'])
    
    return ensemble_model, scaler, explanation_data, ensemble_meta

def load_and_prepare_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('heart.csv')
    return df

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt

def create_feature_distribution(df, feature, explanation_data):
    """Create distribution plot for a feature by target"""
    feature_info = explanation_data['descriptions'][feature]
    feature_name = feature_info['name']
    
    if feature_info['type'] == 'numerical':
        fig = px.histogram(
            df, x=feature, color='target',
            color_discrete_map={0: '#1F77B4', 1: '#FF7F0E'},
            barmode='overlay',
            labels={'target': 'Heart Disease', 0: 'No Disease', 1: 'Disease'},
            title=f'Distribution of {feature_name} by Heart Disease Presence'
        )
        fig.update_layout(height=400)
        return fig
    else:
        # For categorical features
        pivot_df = pd.crosstab(df[feature], df['target'], normalize='index').reset_index()
        pivot_df.columns = [feature, 'No Disease', 'Disease']
        fig = px.bar(
            pivot_df, x=feature, y=['No Disease', 'Disease'],
            title=f'Proportion of Heart Disease by {feature_name}',
            height=400,
            labels={'value': 'Proportion', 'variable': 'Heart Disease Status'}
        )
        return fig

def make_prediction(features, ensemble_model, scaler):
    """Make prediction using the ensemble model"""
    # Convert input to DataFrame
    input_df = pd.DataFrame([features])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Get individual model predictions
    nn_pred = ensemble_model.nn_model.predict(input_scaled).flatten()[0]
    xgb_pred = ensemble_model.xgb_model.predict_proba(input_scaled)[0, 1]
    
    # Blend predictions
    prediction_proba = (ensemble_model.blend_weights[0] * nn_pred + 
                        ensemble_model.blend_weights[1] * xgb_pred)
    
    # Use a custom threshold (0.3) rather than 0.5 as the model may be biased toward positive predictions
    prediction = 1 if prediction_proba >= 0.3 else 0
    
    # Debug information
    st.session_state['debug_info'] = {
        'raw_features': features,
        'scaled_features': input_scaled.tolist(),
        'neural_network_prediction': float(nn_pred),
        'xgboost_prediction': float(xgb_pred),
        'blend_weights': ensemble_model.blend_weights,
        'raw_prediction': float(prediction_proba)
    }
    
    return prediction, prediction_proba

def main():
    # Load resources
    try:
        ensemble_model, scaler, explanation_data, ensemble_meta = load_model_resources()
        df = load_and_prepare_data()
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        st.warning("Please make sure you've run model.py first to train the model and generate explanations.")
        return
    
    # Display header
    st.markdown("<h1 class='main-header'>❤️ Heart Disease Prediction & Explanation</h1>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Feature Importance", "Feature Explorer", "Prediction"])
    
    # Tab 1: Overview
    with tab1:
        st.markdown("<h2 class='sub-header'>About Heart Disease</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='description'>
        Heart disease is a leading cause of death globally. Early detection and prevention are crucial for reducing mortality rates.
        This application uses machine learning to predict the likelihood of heart disease based on various health parameters and explains
        how different factors contribute to heart disease risk.
        </p>
        """, unsafe_allow_html=True)
        
        # Display model performance metrics
        st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='description'>
        This application uses a state-of-the-art ensemble model combining deep neural networks with gradient boosting (XGBoost)
        to achieve superior prediction accuracy and robustness.
        </p>
        """, unsafe_allow_html=True)
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Neural Network Accuracy", 
                value=f"{ensemble_meta['nn_accuracy']:.2%}",
                delta=f"{ensemble_meta['nn_accuracy']-0.85:.2%}"
            )
            
        with col2:
            st.metric(
                label="XGBoost Accuracy", 
                value=f"{ensemble_meta['xgb_accuracy']:.2%}",
                delta=f"{ensemble_meta['xgb_accuracy']-0.85:.2%}"
            )
            
        with col3:
            st.metric(
                label="Ensemble Accuracy", 
                value=f"{ensemble_meta['ensemble_accuracy']:.2%}",
                delta=f"{ensemble_meta['ensemble_accuracy']-0.85:.2%}"
            )
            
        # Create three columns for AUC metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Neural Network AUC", 
                value=f"{ensemble_meta['nn_auc']:.2%}"
            )
            
        with col2:
            st.metric(
                label="XGBoost AUC", 
                value=f"{ensemble_meta['xgb_auc']:.2%}"
            )
            
        with col3:
            st.metric(
                label="Ensemble AUC", 
                value=f"{ensemble_meta['ensemble_auc']:.2%}"
            )
            
        st.markdown("""
        <p class='description'>
        <strong>Model Blend:</strong> The predictions are a weighted blend of neural network and XGBoost outputs, 
        optimized for maximum accuracy and generalization. The ensemble approach reduces prediction variability and 
        improves overall performance.
        </p>
        """, unsafe_allow_html=True)
        
        # Display dataset information
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        st.write(f"**Number of records:** {df.shape[0]}")
        st.write(f"**Number of features:** {df.shape[1] - 1}")
        st.write(f"**Target distribution:** {df['target'].value_counts()[1]} positive cases, {df['target'].value_counts()[0]} negative cases")
        
        # Display feature descriptions
        st.markdown("<h2 class='sub-header'>Feature Descriptions</h2>", unsafe_allow_html=True)
        for feature, info in explanation_data['descriptions'].items():
            st.markdown(f"<h3 class='feature-header'>{info['name']} ({feature})</h3>", unsafe_allow_html=True)
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"<div class='feature-impact'><strong>Impact on Heart Disease:</strong> {info['impact']}</div>", unsafe_allow_html=True)
            st.markdown("---")
        
        # Display correlation heatmap
        st.markdown("<h2 class='sub-header'>Feature Correlations</h2>", unsafe_allow_html=True)
        st.pyplot(create_correlation_heatmap(df))
    
    # Tab 2: Feature Importance
    with tab2:
        st.markdown("<h2 class='sub-header'>Feature Importance Analysis</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='description'>
        Understanding which features have the most significant impact on predicting heart disease is crucial for both model interpretability
        and for identifying key risk factors.
        </p>
        """, unsafe_allow_html=True)
        
        # Display permutation importance
        st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
        st.image('models/feature_importance.png')
        
        st.markdown("<h3>Top 5 Most Important Features</h3>", unsafe_allow_html=True)
        importances = explanation_data['importances']
        for i, (feature, importance) in enumerate(importances[:5]):
            feature_info = explanation_data['descriptions'][feature]
            st.markdown(f"**{i+1}. {feature_info['name']} ({feature})** - Importance Score: {importance:.4f}")
            st.markdown(f"<div class='feature-impact'>{feature_info['impact']}</div>", unsafe_allow_html=True)
        
        # Display SHAP values if available
        if os.path.exists('models/shap_summary.png') and explanation_data.get('shap_success', False):
            st.markdown("<h3>SHAP Values</h3>", unsafe_allow_html=True)
            st.markdown("""
            <p class='description'>
            SHAP (SHapley Additive exPlanations) values help understand the contribution of each feature to individual predictions.
            Red points indicate higher feature values, while blue points indicate lower values. Features are ordered by their overall importance.
            </p>
            """, unsafe_allow_html=True)
            st.image('models/shap_summary.png')
        else:
            st.markdown("<h3>SHAP Values</h3>", unsafe_allow_html=True)
            st.info("""
            SHAP values are not available for this model. This could be due to compatibility issues between the SHAP library 
            and the current TensorFlow version. The model is still fully functional for predictions and feature importance
            is still available above.
            """)
    
    # Tab 3: Feature Explorer
    with tab3:
        st.markdown("<h2 class='sub-header'>Feature Explorer</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='description'>
        Explore how each feature is distributed across patients with and without heart disease. This can help visualize
        the relationship between individual features and the presence of heart disease.
        </p>
        """, unsafe_allow_html=True)
        
        # Feature selector
        feature_options = [(feature_info['name'], feature) for feature, feature_info in explanation_data['descriptions'].items()]
        selected_feature_name, selected_feature = st.selectbox(
            "Select a feature to explore:",
            options=feature_options,
            format_func=lambda x: x[0]
        )
        
        # Show feature distribution
        st.plotly_chart(create_feature_distribution(df, selected_feature, explanation_data))
        
        # Show feature details
        feature_info = explanation_data['descriptions'][selected_feature]
        st.markdown(f"<h3>About {feature_info['name']}</h3>", unsafe_allow_html=True)
        st.markdown(f"**Description:** {feature_info['description']}")
        st.markdown(f"<div class='feature-impact'><strong>Impact on Heart Disease:</strong> {feature_info['impact']}</div>", unsafe_allow_html=True)
        
        # Show feature statistics
        st.markdown("<h3>Feature Statistics</h3>", unsafe_allow_html=True)
        
        feature_stats = explanation_data['feature_stats'][selected_feature]
        if feature_info['type'] == 'numerical':
            stats_df = pd.DataFrame({
                'Statistic': ['Minimum', 'Maximum', 'Mean', 'Median', 'Standard Deviation'],
                'Value': [
                    feature_stats['min'],
                    feature_stats['max'],
                    feature_stats['mean'],
                    feature_stats['median'],
                    feature_stats['std']
                ]
            })
            st.table(stats_df)
        else:
            # For categorical features, show value counts
            value_counts = feature_stats['value_counts']
            values = list(value_counts.keys())
            counts = list(value_counts.values())
            
            # Create a bar chart
            fig = px.bar(
                x=values, y=counts,
                labels={'x': selected_feature, 'y': 'Count'},
                title=f'Distribution of {feature_info["name"]} Values'
            )
            st.plotly_chart(fig)
    
    # Tab 4: Prediction
    with tab4:
        st.markdown("<h2 class='sub-header'>Heart Disease Prediction</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='description'>
        Enter your health parameters below to get a prediction for heart disease risk. This tool is for educational purposes only
        and should not replace professional medical advice.
        </p>
        """, unsafe_allow_html=True)
        
        # Create a form for user input
        with st.form("prediction_form"):
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=20, max_value=100, value=28)
                sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=0)
                cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                             format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x], index=1)
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=110)
                chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=160)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
                restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], 
                                  format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x], index=0)
            
            with col2:
                thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=190)
                exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
                oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
                slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], 
                                format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x], index=0)
                ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4], index=0)
                thal = st.selectbox("Thalassemia", options=[1, 2, 3], 
                               format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x], index=1)
            
            submit_button = st.form_submit_button("Predict Heart Disease Risk")
        
        # Make prediction when the form is submitted
        if submit_button:
            # Collect all features into a dictionary
            features = {
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal
            }
            
            # Make prediction
            prediction, probability = make_prediction(features, ensemble_model, scaler)
            
            # Display result
            st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
            
            # Adjust probability display to provide more nuanced results
            # The model appears to be biased toward high probabilities
            adjusted_probability = min(probability, 0.95)  # Cap the maximum displayed probability at 0.95
            
            # Display additional context for very high probabilities
            probability_note = ""
            if probability > 0.9:
                probability_note = " (Model prediction confidence may be overestimated)"
            
            if prediction == 1:
                st.markdown(f"<div class='prediction-result high-risk'>High Risk of Heart Disease<br>Probability: {adjusted_probability:.2f}{probability_note}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-result low-risk'>Low Risk of Heart Disease<br>Probability: {adjusted_probability:.2f}{probability_note}</div>", unsafe_allow_html=True)
            
            # Add debug information for transparency
            with st.expander("Model Debug Information"):
                if 'debug_info' in st.session_state:
                    st.write("Neural Network Prediction:", st.session_state['debug_info']['neural_network_prediction'])
                    st.write("XGBoost Prediction:", st.session_state['debug_info']['xgboost_prediction'])
                    st.write("Blend Weights:", st.session_state['debug_info']['blend_weights'])
                    st.write("Prediction threshold: 0.3 (adjusted from standard 0.5)")
                    st.write("Note: If raw prediction is consistently high (>0.9) for various inputs, the model may need retraining with more balanced data.")
                    
                    st.markdown("##### Feature Values After Scaling")
                    scaled_features = pd.DataFrame(st.session_state['debug_info']['scaled_features']).T
                    st.write(scaled_features)
            
            # Add risk factors based on the input
            st.markdown("<h3>Risk Factor Analysis</h3>", unsafe_allow_html=True)
            
            # Identify potential risk factors
            risk_factors = []
            
            # Check age
            if age > 50:
                risk_factors.append(f"Age ({age} years) - Age above 50 increases heart disease risk.")
            
            # Check sex
            if sex == 1:
                risk_factors.append("Sex (Male) - Men generally have a higher risk of heart disease than women.")
            
            # Check chest pain
            if cp == 0:
                risk_factors.append("Chest Pain Type (Typical Angina) - Typical angina is associated with coronary artery disease.")
            
            # Check blood pressure
            if trestbps > 140:
                risk_factors.append(f"Resting Blood Pressure ({trestbps} mm Hg) - Blood pressure above 140 mm Hg indicates hypertension.")
            
            # Check cholesterol
            if chol > 240:
                risk_factors.append(f"Serum Cholesterol ({chol} mg/dl) - Cholesterol levels above 240 mg/dl are considered high.")
            
            # Check fasting blood sugar
            if fbs == 1:
                risk_factors.append("Fasting Blood Sugar (>120 mg/dl) - Elevated blood sugar may indicate diabetes or prediabetes.")
            
            # Check exercise induced angina
            if exang == 1:
                risk_factors.append("Exercise Induced Angina (Yes) - Chest pain during exercise can indicate coronary artery disease.")
            
            # Check ST depression
            if oldpeak > 1.0:
                risk_factors.append(f"ST Depression ({oldpeak}) - ST depression greater than 1.0 may indicate ischemia.")
            
            # Check number of vessels
            if ca > 0:
                risk_factors.append(f"Major Vessels ({ca}) - Having {ca} colored vessels may indicate coronary artery disease.")
            
            # Check thalassemia
            if thal == 3:
                risk_factors.append("Thalassemia (Reversible Defect) - Reversible defects are associated with ischemic heart disease.")
            
            # Display risk factors
            if risk_factors:
                st.markdown("**Potential risk factors based on your inputs:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("**No major risk factors identified based on your inputs.**")
            
            st.markdown("<p class='footnote'>Note: This prediction is based on a machine learning model and should not replace professional medical advice. If you have concerns about your heart health, please consult a healthcare provider.</p>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("<p class='footnote'>© 2023 Heart Disease Predictor Application | Powered by TensorFlow & Streamlit | Deployed on Hugging Face Spaces</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 