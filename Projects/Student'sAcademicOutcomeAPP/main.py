import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title=" Student's Academic Outcome", layout="wide")

st.title("🎓 STUDENT'S ACADEMIC OUTCOME | Prediction APP")

# Carga de datos (esto se puede cambiar por tu CSV real)
@st.cache_data
def load_data():
    df = pd.read_csv("students_dropout_academic_success.csv")
    return df

# Entrenamiento básico del modelo y selección de top features
@st.cache_resource
def train_model(df):
    df = df.copy()
    df['target'] = df['target'].astype(str)
    target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    df['target'] = df['target'].map(target_map)

    categorical_cols = df.select_dtypes(include='object').columns
    if 'target' in categorical_cols:
        categorical_cols = categorical_cols.drop('target')

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X_full = df.drop('target', axis=1)
    y = df['target']

    temp_model = RandomForestClassifier(random_state=42)
    temp_model.fit(X_full, y)
    importances = temp_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_full.columns, 'Importance': importances})
    top_features_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

    # Entrenar modelo solo con las top 10 features
    top_features = top_features_df['Feature'].tolist()
    X = X_full[top_features]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, top_features_df

# PESTAÑAS PRINCIPALES
tabs = st.tabs(["📄 INFO", "📋 PREDICT"])

# TAB 1 - INFORMACIÓN DEL PROYECTO
with tabs[0]:
    st.header("About the Project")
    st.markdown("""
This app predicts whether a student will:

- ❌ Drop out
- ⏳ Stay enrolled
- ✅ Graduate

Based on academic, economic, and demographic variables.
""")

# TAB 2 - FORMULARIO DE PREDICCIÓN
with tabs[1]:
    st.header("SIMULATES A STUDENT PROFILE")

    df = load_data()
    model, top_features_df = train_model(df)
    feature_names = top_features_df['Feature'].tolist()

    user_input = {}
    with st.form("prediction_form"):
        label_map = {
            "Curricular units 2nd sem (approved)": "Subjects Passed (2nd Semester)",
            "Curricular units 2nd sem (grade)": "Grade Average (2nd Semester)",
            "Curricular units 2nd sem (evaluations)": "Total Evaluations (2nd Semester)",
            "Curricular units 1st sem (approved)": "Subjects Passed (1st Semester)",
            "Curricular units 1st sem (grade)": "Grade Average (1st Semester)",
            "Curricular units 1st sem (evaluations)": "Total Evaluations (1st Semester)",
            "Previous qualification (grade)": "Previous Qualification Grade"
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
                    "Admission grade": (0, 200),
                    "Grade Average (2nd Semester)": (0, 10),
                    "Grade Average (1st Semester)": (0, 10),
                    "Subjects Passed (2nd Semester)": (0, 20),
                    "Subjects Passed (1st Semester)": (0, 20),
                    "Age at enrollment": (17, 40),
                    "Total Evaluations (2nd Semester)": (0, 40),
                    "Total Evaluations (1st Semester)": (0, 40),
                    "Previous Qualification Grade": (0, 10)
                    
                }
                if label in custom_limits:
                    min_val, max_val = custom_limits[label]
                else:
                    min_val = int(df[feature].min())
                    max_val = int(df[feature].max())
                val = st.number_input(f"{label}", min_value=min_val, max_value=max_val, value=min_val)

            user_input[feature] = val
        submitted = st.form_submit_button("Predict")

    if submitted:
        X_input = pd.DataFrame([user_input])[feature_names]
        pred = model.predict(X_input)[0]
        label_map = {0: '❌ Dropout', 1: '⏳ Enrolled', 2: '✅ Graduate'}
        st.success(f"Prediction: {label_map[pred]}")














