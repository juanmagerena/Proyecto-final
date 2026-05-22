import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Predicción de Riesgo Cardíaco",
    page_icon="🫀",
    layout="wide"
)

# ── Carga de artefactos ──────────────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model  = joblib.load(os.path.join(BASE_PATH, "best_heart_attack_prediction_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_PATH, "scaler.joblib"))
    # Features en el orden exacto que el modelo y el scaler esperan
    feature_names = [
        'BMI', 'Triglycerides', 'Exercise Hours Per Week',
        'Sedentary Hours Per Day', 'Cholesterol', 'Systolic_BP',
        'Heart Rate', 'Age', 'Diastolic_BP', 'Stress Level',
        'Physical Activity Days Per Week', 'Sleep Hours Per Day'
    ]
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Error al cargar los artefactos: {e}")
    st.stop()

# ── Metadatos de cada feature ────────────────────────────────────────────────
FEATURE_META = {
    'BMI':                            {"label": "Índice de Masa Corporal (BMI)",           "min": 15.0, "max": 50.0, "default": 25.0, "step": 0.1},
    'Triglycerides':                  {"label": "Triglicéridos (mg/dL)",                   "min": 30,   "max": 800,  "default": 150,  "step": 1},
    'Exercise Hours Per Week':        {"label": "Horas de ejercicio por semana",            "min": 0.0,  "max": 20.0, "default": 5.0,  "step": 0.1},
    'Sedentary Hours Per Day':        {"label": "Horas sedentarias por día",                "min": 0.0,  "max": 12.0, "default": 6.0,  "step": 0.1},
    'Cholesterol':                    {"label": "Colesterol (mg/dL)",                       "min": 120,  "max": 400,  "default": 200,  "step": 1},
    'Systolic_BP':                    {"label": "Presión arterial sistólica (mmHg)",         "min": 90,   "max": 200,  "default": 120,  "step": 1},
    'Heart Rate':                     {"label": "Frecuencia cardíaca (bpm)",                "min": 40,   "max": 110,  "default": 70,   "step": 1},
    'Age':                            {"label": "Edad",                                     "min": 18,   "max": 90,   "default": 50,   "step": 1},
    'Diastolic_BP':                   {"label": "Presión arterial diastólica (mmHg)",        "min": 60,   "max": 110,  "default": 80,   "step": 1},
    'Stress Level':                   {"label": "Nivel de estrés (1–10)",                   "min": 1,    "max": 10,   "default": 5,    "step": 1},
    'Physical Activity Days Per Week':{"label": "Días de actividad física por semana",      "min": 0,    "max": 7,    "default": 3,    "step": 1},
    'Sleep Hours Per Day':            {"label": "Horas de sueño por día",                   "min": 3,    "max": 12,   "default": 7,    "step": 1},
}

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("🩺 Datos del Paciente")
st.sidebar.info("Ajusta los valores y presiona **Predecir**.")

input_data = {}
for feat in feature_names:
    m = FEATURE_META[feat]
    input_data[feat] = st.sidebar.slider(
        m["label"],
        min_value=m["min"], max_value=m["max"],
        value=m["default"], step=m["step"]
    )

# ── Cuerpo ───────────────────────────────────────────────────────────────────
st.title("🫀 Predicción de Riesgo de Ataque Cardíaco")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Datos ingresados")
    input_df = pd.DataFrame([input_data])[feature_names]
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("Resultado")
    if st.button("🔍 Predecir", use_container_width=True):
        try:
            input_scaled    = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

            prediction       = model.predict(input_scaled_df)[0]
            prediction_proba = model.predict_proba(input_scaled_df)[0]

            prob_riesgo = prediction_proba[1] * 100
            prob_no     = prediction_proba[0] * 100

            if prediction == 1:
                st.error("⚠️ **ALTO RIESGO** de ataque cardíaco")
            else:
                st.success("✅ **BAJO RIESGO** de ataque cardíaco")

            st.metric("Probabilidad de riesgo",  f"{prob_riesgo:.1f}%")
            st.metric("Probabilidad sin riesgo",  f"{prob_no:.1f}%")
            st.progress(int(prob_riesgo))

        except Exception as e:
            st.error(f"Error al predecir: {e}")
    else:
        st.info("Ajusta los valores y haz clic en **Predecir**.")
