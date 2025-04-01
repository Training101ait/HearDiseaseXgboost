import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.optimizers import Adam
import xgboost as xgb

# Create directory to store model
os.makedirs('models', exist_ok=True)

def feature_descriptions():
    """Provide detailed descriptions of each feature"""
    descriptions = {
        'age': {
            'name': 'Age',
            'description': 'Age of the patient in years',
            'impact': 'Increasing age is associated with a higher risk of heart disease due to the gradual deterioration of cardiovascular function over time.',
            'type': 'numerical'
        },
        'sex': {
            'name': 'Sex',
            'description': 'Gender of the patient (1 = male, 0 = female)',
            'impact': 'Men are generally at higher risk of heart disease than women, especially before women reach menopause.',
            'type': 'categorical'
        },
        'cp': {
            'name': 'Chest Pain Type',
            'description': 'Type of chest pain (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)',
            'impact': 'The type of chest pain can be a strong indicator of heart disease, with typical angina often associated with coronary artery disease.',
            'type': 'categorical'
        },
        'trestbps': {
            'name': 'Resting Blood Pressure',
            'description': 'Resting blood pressure in mm Hg on admission to the hospital',
            'impact': 'High resting blood pressure is a major risk factor for heart disease as it can damage artery walls and lead to atherosclerosis.',
            'type': 'numerical'
        },
        'chol': {
            'name': 'Serum Cholesterol',
            'description': 'Serum cholesterol in mg/dl',
            'impact': 'Elevated cholesterol levels can lead to the formation of plaques in the arteries, increasing the risk of heart disease.',
            'type': 'numerical'
        },
        'fbs': {
            'name': 'Fasting Blood Sugar',
            'description': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
            'impact': 'Elevated blood sugar levels, particularly in diabetics, increase the risk of heart disease by damaging blood vessels and accelerating atherosclerosis.',
            'type': 'categorical'
        },
        'restecg': {
            'name': 'Resting ECG Results',
            'description': 'Resting electrocardiographic results (0: Normal, 1: Having ST-T wave abnormality, 2: Showing probable or definite left ventricular hypertrophy)',
            'impact': 'Abnormal ECG results can indicate existing heart problems, which are associated with a higher risk of heart disease.',
            'type': 'categorical'
        },
        'thalach': {
            'name': 'Maximum Heart Rate',
            'description': 'Maximum heart rate achieved during exercise',
            'impact': 'A lower maximum heart rate during exercise may indicate decreased cardiovascular fitness and potential heart problems.',
            'type': 'numerical'
        },
        'exang': {
            'name': 'Exercise Induced Angina',
            'description': 'Exercise induced angina (1 = yes, 0 = no)',
            'impact': 'Chest pain during exercise can indicate that the heart is not receiving enough oxygen, often due to coronary artery disease.',
            'type': 'categorical'
        },
        'oldpeak': {
            'name': 'ST Depression',
            'description': 'ST depression induced by exercise relative to rest',
            'impact': 'Greater ST depression during exercise indicates potential issues with heart muscle oxygenation and is associated with heart disease.',
            'type': 'numerical'
        },
        'slope': {
            'name': 'Slope of Peak Exercise ST Segment',
            'description': 'The slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)',
            'impact': 'Downsloping or flat ST segments are more associated with heart disease than upsloping segments.',
            'type': 'categorical'
        },
        'ca': {
            'name': 'Number of Major Vessels',
            'description': 'Number of major vessels (0-4) colored by fluoroscopy',
            'impact': 'More major vessels with calcium deposits or blockages indicate more advanced coronary artery disease.',
            'type': 'categorical'
        },
        'thal': {
            'name': 'Thalassemia',
            'description': 'Thalassemia (1: Normal, 2: Fixed defect, 3: Reversible defect)',
            'impact': 'Thalassemia, especially reversible defects, can indicate issues with heart blood flow and is associated with heart disease.',
            'type': 'categorical'
        }
    }
    return descriptions

def calculate_feature_importance(model, X_scaled, y, feature_names):
    """Calculate feature importance using a custom approach for TensorFlow models"""
    print("Calculating feature importance...")
    
    # Initialize importance scores dictionary
    importances = {}
    
    # Baseline accuracy
    baseline_pred = model.predict(X_scaled)
    baseline_pred_classes = (baseline_pred > 0.5).astype(int).flatten()
    baseline_accuracy = np.mean(baseline_pred_classes == y)
    
    # Calculate importance for each feature
    for i, feature in enumerate(feature_names):
        # Create a perturbed version of the data
        X_perturbed = X_scaled.copy()
        # Shuffle the values of the current feature to break any correlation
        X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])
        
        # Make predictions with perturbed data
        perturbed_pred = model.predict(X_perturbed)
        perturbed_pred_classes = (perturbed_pred > 0.5).astype(int).flatten()
        perturbed_accuracy = np.mean(perturbed_pred_classes == y)
        
        # Importance is the decrease in accuracy when the feature is permuted
        importances[feature] = baseline_accuracy - perturbed_accuracy
    
    # Convert to sorted list of tuples
    importance_values = [(feature, score) for feature, score in importances.items()]
    importance_values.sort(key=lambda x: x[1], reverse=True)
    
    return importance_values

def plot_feature_importance(importance_values, save_path='models/feature_importance.png'):
    """Plot and save feature importance"""
    features, scores = zip(*importance_values)
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': scores
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance')
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance (Permutation Importance)')
    plt.xlabel('Mean Decrease in Accuracy')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return importance_values

def calculate_shap_values(model, X_scaled, X, save_path='models/shap_summary.png'):
    """Calculate and visualize SHAP values using KernelExplainer instead of DeepExplainer"""
    try:
        print("Calculating SHAP values...")
        
        # Sample a smaller subset of data for efficiency (100 samples)
        sample_indices = np.random.choice(len(X_scaled), size=min(100, len(X_scaled)), replace=False)
        X_sample = X_scaled[sample_indices]
        
        # Define prediction function that returns probabilities
        def model_predict(x):
            return model.predict(x)
        
        # Use KernelExplainer instead of DeepExplainer
        explainer = shap.KernelExplainer(model_predict, shap.kmeans(X_sample, 10))
        
        # Calculate SHAP values (use a small subset for speed)
        shap_values = explainer.shap_values(X_sample)
        
        # If shap_values is a list, take the first element (for binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, pd.DataFrame(X_sample, columns=X.columns), show=False)
        plt.savefig(save_path)
        plt.close()
        
        return shap_values
    except Exception as e:
        print(f"SHAP calculation error: {e}")
        return None

def prepare_explanation_data(df, X, X_scaled, y, model, feature_names):
    """Prepare comprehensive explanation data for the Streamlit app"""
    descriptions = feature_descriptions()
    
    # Calculate importances
    importance_values = calculate_feature_importance(model, X_scaled, y, feature_names)
    sorted_importances = plot_feature_importance(importance_values)
    
    # Calculate SHAP values with error handling
    try:
        print("Attempting to calculate SHAP values...")
        shap_values = calculate_shap_values(model, X_scaled, X)
        shap_success = shap_values is not None
    except Exception as e:
        print(f"SHAP calculation completely failed: {e}")
        print("Continuing without SHAP values")
        shap_success = False
    
    # Prepare feature stats
    feature_stats = {}
    for col in X.columns:
        if descriptions[col]['type'] == 'numerical':
            feature_stats[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            }
        else:
            feature_stats[col] = {
                'value_counts': df[col].value_counts().to_dict()
            }
    
    # Correlations with target
    correlations = {}
    for col in X.columns:
        correlations[col] = df[col].corr(df['target'])
    
    # Save all explanation data
    explanation_data = {
        'descriptions': descriptions,
        'importances': sorted_importances,
        'feature_stats': feature_stats,
        'correlations': correlations,
        'shap_success': shap_success
    }
    
    # Save to file
    joblib.dump(explanation_data, 'models/explanation_data.pkl')
    
    return explanation_data

def build_neural_network(input_shape):
    """Build a deep neural network model with better generalization capabilities"""
    inputs = Input(shape=(input_shape,))
    
    # First block - smaller initial layer with stronger regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Increase dropout to prevent memorization
    
    # Second block with skip connection
    block_1 = x
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Increase dropout
    
    # Third block
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Increase dropout
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0003),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_xgboost_model():
    """Build a more conservative XGBoost model to prevent overfitting"""
    return xgb.XGBClassifier(
        n_estimators=200,          # Fewer estimators
        max_depth=3,               # Reduced depth
        learning_rate=0.005,       # Lower learning rate
        subsample=0.7,             # More aggressive subsampling
        colsample_bytree=0.7,      # More aggressive column sampling
        min_child_weight=2,        # Increased to prevent overfitting
        gamma=0.2,                 # Increased regularization
        reg_alpha=0.2,             # Increased L1 regularization
        reg_lambda=2,              # Increased L2 regularization
        objective='binary:logistic',
        eval_metric='auc'
    )

def blend_predictions(nn_preds, xgb_preds, blend_weights=None):
    """Blend predictions from different models"""
    if blend_weights is None:
        # Default weights (can be tuned)
        blend_weights = [0.5, 0.5]
    
    return (blend_weights[0] * nn_preds + blend_weights[1] * xgb_preds)

# Main execution
print("Loading and preparing data...")
# Load the dataset
df = pd.read_csv('heart.csv')

# Display basic information about the data
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nColumns:")
for col in df.columns:
    print(f"- {col}")

# Data preprocessing
X = df.drop('target', axis=1)
y = df['target']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, 'models/scaler.pkl')

# Use stratified shuffle split to ensure balanced data distribution
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_val_idx, test_idx in sss.split(X_scaled, y):
    X_train_val, X_test = X_scaled[train_val_idx], X_scaled[test_idx]
    y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

print("\n------ TRAINING ENSEMBLE OF MODELS ------")
print("1. Training Neural Network with Cross-Validation...")

# Cross-validation setup
n_folds = 5
kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Neural Network training with cross-validation
nn_val_scores = []
nn_models = []

# Common callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10,  # Reduced patience for faster stopping
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  # Larger reduction factor
    patience=5,   # Reduced patience
    min_lr=0.00001,
    verbose=1
)

fold = 1
for train_idx, val_idx in kfold.split(X_train_val, y_train_val):
    print(f"\nTraining fold {fold}/{n_folds}")
    
    # Split data
    X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx]
    y_train_fold, y_val_fold = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    
    # Build and train neural network
    nn_model = build_neural_network(X_train_fold.shape[1])
    
    history = nn_model.fit(
        X_train_fold, y_train_fold,
        epochs=100,
        batch_size=16,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=[early_stopping, reduce_lr],
        verbose=2
    )
    
    # Evaluate on validation set
    val_loss, val_acc = nn_model.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f"Fold {fold} - Validation Accuracy: {val_acc:.4f}")
    
    nn_val_scores.append(val_acc)
    nn_models.append(nn_model)
    
    fold += 1

print(f"\nNeural Network CV Accuracy: {np.mean(nn_val_scores):.4f} ± {np.std(nn_val_scores):.4f}")

# Train XGBoost model with cross-validation
print("\n2. Training XGBoost Model with Cross-Validation...")
xgb_model = build_xgboost_model()

xgb_val_preds = np.zeros(len(X_train_val))
xgb_val_scores = []

fold = 1
for train_idx, val_idx in kfold.split(X_train_val, y_train_val):
    # Split data
    X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx]
    y_train_fold, y_val_fold = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    
    # Train XGBoost model
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=0
    )
    
    # Get predictions
    xgb_val_preds_fold = xgb_model.predict_proba(X_val_fold)[:, 1]
    xgb_val_preds[val_idx] = xgb_val_preds_fold
    
    # Calculate accuracy
    preds_binary = (xgb_val_preds_fold > 0.5).astype(int)
    acc = accuracy_score(y_val_fold, preds_binary)
    xgb_val_scores.append(acc)
    
    print(f"Fold {fold} - XGBoost Validation Accuracy: {acc:.4f}")
    fold += 1

print(f"\nXGBoost CV Accuracy: {np.mean(xgb_val_scores):.4f} ± {np.std(xgb_val_scores):.4f}")

# Train final models on all training data
print("\n3. Training final models on all training data...")

# Train Neural Network on all training data
final_nn_model = build_neural_network(X_train_val.shape[1])
final_nn_model.fit(
    X_train_val, y_train_val,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# Train XGBoost on all training data
final_xgb_model = build_xgboost_model()
final_xgb_model.fit(
    X_train_val, y_train_val,
    eval_set=[(X_train_val, y_train_val)],
    verbose=0
)

# Save the XGBoost model
joblib.dump(final_xgb_model, 'models/xgboost_model.pkl')

# Evaluate on test set
print("\n------ FINAL MODEL EVALUATION ------")

# Neural Network predictions
nn_test_preds = final_nn_model.predict(X_test).flatten()
nn_test_binary = (nn_test_preds > 0.5).astype(int)
nn_accuracy = accuracy_score(y_test, nn_test_binary)
nn_auc = roc_auc_score(y_test, nn_test_preds)

print(f"Neural Network Test Accuracy: {nn_accuracy:.4f}")
print(f"Neural Network Test AUC: {nn_auc:.4f}")

# XGBoost predictions
xgb_test_preds = final_xgb_model.predict_proba(X_test)[:, 1]
xgb_test_binary = (xgb_test_preds > 0.5).astype(int)
xgb_accuracy = accuracy_score(y_test, xgb_test_binary)
xgb_auc = roc_auc_score(y_test, xgb_test_preds)

print(f"XGBoost Test Accuracy: {xgb_accuracy:.4f}")
print(f"XGBoost Test AUC: {xgb_auc:.4f}")

# Blend predictions (ensemble)
blend_weights = [0.6, 0.4]  # Weighted toward neural network
ensemble_test_preds = blend_predictions(nn_test_preds, xgb_test_preds, blend_weights)
ensemble_test_binary = (ensemble_test_preds > 0.5).astype(int)
ensemble_accuracy = accuracy_score(y_test, ensemble_test_binary)
ensemble_auc = roc_auc_score(y_test, ensemble_test_preds)

print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
print(f"Ensemble Test AUC: {ensemble_auc:.4f}")

# Create a prediction wrapper class for the ensemble
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
ensemble_model = EnsembleModel(final_nn_model, final_xgb_model, blend_weights)

# Save the neural network model
final_nn_model.save('models/heart_model.keras')

# Save the ensemble metadata
ensemble_meta = {
    'blend_weights': blend_weights,
    'nn_accuracy': nn_accuracy,
    'xgb_accuracy': xgb_accuracy,
    'ensemble_accuracy': ensemble_accuracy,
    'nn_auc': nn_auc,
    'xgb_auc': xgb_auc,
    'ensemble_auc': ensemble_auc
}
joblib.dump(ensemble_meta, 'models/ensemble_meta.pkl')

# Save feature names for feature importance analysis
feature_names = X.columns.tolist()
np.save('models/feature_names.npy', feature_names)

# Plot training history for neural network
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('models/training_history.png')

print("Model training complete and saved successfully!")

# Generate feature importance and explanation data using the ensemble model
print("Generating feature importance and explanations...")
explanation_data = prepare_explanation_data(df, X, X_scaled, y, ensemble_model, feature_names)

# Print top important features
print(f"\nTop 5 most important features:")
for feature, importance in explanation_data['importances'][:5]:
    print(f"- {feature}: {importance:.4f}")

print("\nAll processing complete! You can now run the Streamlit app with: streamlit run app.py") 