
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os # Import os for debugging path

# Load the trained model
# The model is expected to be in the same directory as this script.
try:
    model = joblib.load('best_heart_attack_prediction_model.pkl')
    st.success("Model 'best_heart_attack_prediction_model.pkl' loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure 'best_heart_attack_prediction_model.pkl' is in the current directory (current: {os.getcwd()}).")
    st.stop()

# Define feature names and their ranges/options based on a typical heart disease dataset
# These ranges are approximations and might need adjustment based on the actual training data.
features_info = {
    'age': {'type': 'slider', 'min': 29, 'max': 77, 'default': 50, 'step': 1},
    'sex': {'type': 'radio', 'options': {0: 'Female', 1: 'Male'}, 'default': 1}, # 0=female, 1=male
    'cp': {'type': 'select', 'options': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}, 'default': 0},
    'trestbps': {'type': 'slider', 'min': 90, 'max': 200, 'default': 120, 'step': 1}, # Resting blood pressure
    'chol': {'type': 'slider', 'min': 100, 'max': 600, 'default': 200, 'step': 1}, # Serum cholestoral in mg/dl
    'fbs': {'type': 'radio', 'options': {0: '<= 120 mg/dl', 1: '> 120 mg/dl'}, 'default': 0}, # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    'restecg': {'type': 'select', 'options': {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}, 'default': 0}, # Resting electrocardiographic results
    'thalach': {'type': 'slider', 'min': 70, 'max': 220, 'default': 150, 'step': 1}, # Maximum heart rate achieved
    'exang': {'type': 'radio', 'options': {0: 'No', 1: 'Yes'}, 'default': 0}, # Exercise induced angina (1 = yes; 0 = no)
    'oldpeak': {'type': 'slider', 'min': 0.0, 'max': 6.2, 'default': 1.0, 'step': 0.1}, # ST depression induced by exercise relative to rest
    'slope': {'type': 'select', 'options': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}, 'default': 1}, # The slope of the peak exercise ST segment
    'ca': {'type': 'slider', 'min': 0, 'max': 4, 'default': 0, 'step': 1}, # Number of major vessels (0-3) colored by flourosopy (some datasets have 4 as 'missing')
    'thal': {'type': 'select', 'options': {1: 'Normal', 2: 'Fixed defect', 3: 'Reversible defect'}, 'default': 2} # Thalassemia
}

st.set_page_config(page_title="Heart Attack Risk Prediction", layout="wide")

st.title("Heart Attack Risk Prediction App")
st.markdown("Enter patient details to predict the risk of heart attack.")

# Sidebar for user input
st.sidebar.header("Patient Input Features")
st.sidebar.markdown("**Información:** Favor de llenar todos los campos para obtener una predicción precisa.")

input_data = {}
for feature, info in features_info.items():
    if info['type'] == 'slider':
        input_data[feature] = st.sidebar.slider(f"{feature.replace('_', ' ').title()}", info['min'], info['max'], info['default'], info['step'])
    elif info['type'] == 'radio':
        display_options = list(info['options'].values())
        selected_display = st.sidebar.radio(f"{feature.replace('_', ' ').title()}", display_options, index=list(info['options'].keys()).index(info['default']))
        # Map display value back to numerical key
        input_data[feature] = [k for k, v in info['options'].items() if v == selected_display][0]
    elif info['type'] == 'select':
        display_options = list(info['options'].values())
        selected_display = st.sidebar.selectbox(f"{feature.replace('_', ' ').title()}", display_options, index=list(info['options'].keys()).index(info['default']))
        # Map display value back to numerical key
        input_data[feature] = [k for k, v in info['options'].items() if v == selected_display][0]

# Convert input_data to a DataFrame
input_df = pd.DataFrame([input_data])

st.subheader("User Input Data")
st.write(input_df)

# IMPORTANT: Scaling Warning
st.warning("Note: The model was likely trained on *scaled* data. This app assumes the `best_heart_attack_prediction_model.pkl` is capable of handling raw, unscaled inputs directly, or that the scaling step is embedded within the model pipeline itself. If not, the predictions might be inaccurate. For precise predictions, a pre-trained scaler (e.g., StandardScaler or MinMaxScaler) should be loaded and applied to the input data before prediction.")

# Prediction
if st.button("Predict Heart Attack Risk"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("Based on the input, there is a **HIGH RISK** of heart attack.")
        else:
            st.success("Based on the input, there is a **LOW RISK** of heart attack.")

        st.write(f"Probability of Heart Attack: {prediction_proba[0][1]*100:.2f}%")
        st.write(f"Probability of No Heart Attack: {prediction_proba[0][0]*100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure the input features match the model's expected format and types.")

