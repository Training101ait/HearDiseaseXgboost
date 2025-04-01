# Heart Disease Prediction Model

An advanced machine learning application that predicts heart disease risk using an ensemble of neural networks and gradient boosting models.

## Overview

This application analyzes various health parameters to predict the likelihood of heart disease and explains how different factors contribute to heart disease risk. It uses a state-of-the-art ensemble model combining deep neural networks with XGBoost for superior prediction accuracy.

## Features

- Heart disease risk prediction with probability estimates
- Feature importance visualization and explanation
- Interactive data exploration
- Model performance metrics display
- Risk factor analysis based on user inputs

## Configuration

### Environment Setup

1. Python 3.7+ is required
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### File Structure

- `model.py` - Model training script with feature importance analysis
- `app.py` - Streamlit application
- `requirements.txt` - Project dependencies
- `heart.csv` - Dataset (Cleveland Heart Disease dataset)
- `models/` - Directory for saved models and analysis results
- `deploy_to_hf.py` - Script for deploying to Hugging Face Spaces

### Configuration Files

- `.streamlit/config.toml` - Streamlit configuration

Create this file with the following content:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
textColor = "#212529"
font = "sans serif"

[server]
enableCORS = false
enableXsrfProtection = false
```

## Usage

### Training the Model

```
python model.py
```

This will:
1. Load and preprocess the heart disease dataset
2. Train an ensemble of neural network and XGBoost models
3. Save models and feature importance analysis
4. Generate visualizations

### Running the Application

```
streamlit run app.py
```

## Deployment

### Hugging Face Spaces Deployment

To deploy to Hugging Face Spaces:

1. Create a Hugging Face account
2. Install git and the Hugging Face CLI:
   ```
   pip install huggingface_hub
   ```
3. Login to Hugging Face:
   ```
   huggingface-cli login
   ```
4. Deploy using the provided script:
   ```
   python deploy_to_hf.py
   ```

### Hugging Face Spaces Information

When deploying to Hugging Face Spaces, the app will have the following features:

- **Theme**: Custom Streamlit theme with heart disease related color scheme
- **Model**: Ensemble of neural network and XGBoost models
- **Data**: Preprocessed heart disease dataset with standardized features

Note that Hugging Face deployment requires a properly configured `.streamlit/config.toml` file, which is already provided in this repository.

## Model Architecture

The application uses an ensemble of:
- A deep neural network with batch normalization and dropout
- An XGBoost gradient boosting model

The predictions are combined with optimized weights to maximize accuracy.

## Dataset

The dataset includes the following features:
- age: Age in years
- sex: Sex (1 = male, 0 = female)
- cp: Chest pain type (0-3)
- trestbps: Resting blood pressure (mm Hg)
- chol: Serum cholesterol (mg/dl)
- fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- restecg: Resting ECG results (0-2)
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina (1 = yes, 0 = no)
- oldpeak: ST depression induced by exercise
- slope: Slope of peak exercise ST segment (0-2)
- ca: Number of major vessels colored by fluoroscopy (0-4)
- thal: Thalassemia (1-3)
- target: Heart disease diagnosis (1 = present, 0 = absent)

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original heart disease dataset from UCI Machine Learning Repository
- TensorFlow team for the deep learning framework
- Streamlit team for the easy-to-use web app framework 