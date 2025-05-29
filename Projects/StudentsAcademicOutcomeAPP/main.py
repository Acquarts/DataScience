import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Student's Academic Outcome", layout="wide")

st.title("🎓 STUDENT'S ACADEMIC OUTCOME | Prediction APP")

# Cargar modelo ya entrenado
def load_model():
    return joblib.load(os.path.join(os.path.dirname(__file__), "Model_XGBoost_Classifier.pkl"))

# Cargar dataframe de referencia (estructura y límites)
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "df_final.csv"))
    return df

# Selección de idioma
lang = st.radio("Select language / Selecciona idioma", ["English", "Español"], horizontal=True)

# Pestañas
if lang == "English":
    tabs = st.tabs(["📄 Info", "📋 Predict"])
else:
    tabs = st.tabs(["📄 Información", "📋 Predecir"])

# TAB 1 - INFORMACIÓN DEL PROYECTO
with tabs[0]:
    st.header("About the Project" if lang == "English" else "Sobre el Proyecto")
    st.markdown("""
This app predicts whether a student will:

- ❌ Drop out
- ⏳ Stay enrolled
- ✅ Graduate

Based on academic, economic, and demographic variables.
""" if lang == "English" else """
Esta app predice si un estudiante:

- ❌ Abandonará los estudios
- ⏳ Seguirá matriculado
- ✅ Se graduará

Basado en variables académicas, económicas y demográficas.
""")

# TAB 2 - FORMULARIO DE PREDICCIÓN
with tabs[1]:
    st.header("Simulate a Student Profile" if lang == "English" else "Simula un Perfil de Estudiante")

    model = load_model()
    df = load_data()
    feature_names = df.drop("target", axis=1).columns.tolist()

    user_input = {}
    with st.form("prediction_form"):
        label_map = {
            "Curricular units 2nd sem (credited)": "Subjects Passed (2nd Semester)" if lang == "English" else "Asignaturas Aprobadas (2º Semestre)",
            "Curricular units 2nd sem (grade)": "Grade Average (2nd Semester)" if lang == "English" else "Nota Media (2º Semestre)",
            "Curricular units 2nd sem (evaluations)": "Total Evaluations (2nd Semester)" if lang == "English" else "Evaluaciones Totales (2º Semestre)",
            "Previous qualification (grade)": "Previous Qualification Grade" if lang == "English" else "Nota en Estudios Previos",
            "Admission grade": "Admission grade" if lang == "English" else "Nota de Admisión",
            "Tuition fees up to date": "Tuition fees up to date" if lang == "English" else "Matriculaciones hasta la fecha",
            "Age at enrollment": "Age at enrollment" if lang == "English" else "Edad al inscribirse"
        }

        for feature in feature_names:
            label = label_map.get(feature, feature)
            unique_vals = df[feature].unique()

            if df[feature].nunique() == 2:
                options = sorted(unique_vals.tolist())
                val = st.selectbox(f"{label}", options)
            elif df[feature].nunique() <= 10:
                options = sorted(unique_vals.tolist())
                val = st.selectbox(f"{label}", options)
            else:
                custom_limits = {
                    "Admission grade": (100, 150),
                    "Grade Average (2nd Semester)": (0, 20),
                    "Grade Average (1st Semester)": (0, 20),
                    "Subjects Passed (2nd Semester)": (0, 10),
                    "Subjects Passed (1st Semester)": (0, 10),
                    "Age at enrollment": (17, 22),
                    "Total Evaluations (2nd Semester)": (0, 40),
                    "Total Evaluations (1st Semester)": (0, 40),
                    "Previous Qualification Grade": (0, 20),

                    "Nota de Admisión": (100, 150),
                    "Nota Media (2º Semestre)": (0, 20),
                    "Nota Media (1º Semestre)": (0, 20),
                    "Asignaturas Aprobadas (2º Semestre)": (0, 10),
                    "Asignaturas Aprobadas (1º Semestre)": (0, 10),
                    "Edad al inscribirse": (17, 22),
                    "Evaluaciones Totales (2º Semestre)": (0, 40),
                    "Evaluaciones Totales (1º Semestre)": (0, 40),
                    "Nota en Estudios Previos": (0, 20)
                }
                if label in custom_limits:
                    min_val, max_val = custom_limits[label]
                else:
                    min_val = int(df[feature].min())
                    max_val = int(df[feature].max())
                val = st.number_input(f"{label}", min_value=min_val, max_value=max_val, value=min_val)

            user_input[feature] = val
        submitted = st.form_submit_button("Predict" if lang == "English" else "Predecir")

    if submitted:
        X_input = pd.DataFrame([user_input])[feature_names]
        pred = model.predict(X_input)[0]
        label_map = {0: '❌ Dropout', 1: '⏳ Enrolled', 2: '✅ Graduate'}
        st.success(f"Prediction: {label_map[pred]}" if lang == "English" else f"Predicción: {label_map[pred]}")
