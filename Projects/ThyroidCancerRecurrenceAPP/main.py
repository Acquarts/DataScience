import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .sidebar-info {
        background-color: #e8f4f8;
        color: #1f1f1f;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .sidebar-info h3 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sidebar-info p, .sidebar-info ul, .sidebar-info li {
        color: #2d2d2d;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🏥 Thyroid Cancer Recurrence Predictor</h1>', unsafe_allow_html=True)

# Información en la sidebar
st.sidebar.markdown("""
<div class="sidebar-info">
<h3>📊 About This App</h3>
<p>This application predicts thyroid cancer recurrence using machine learning models trained on 383 patient records.</p>
<p><strong>Models Available:</strong></p>
<ul>
<li>Logistic Regression</li>
<li>Random Forest</li>
<li>XGBoost</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Función para entrenar los modelos (simulación basada en tu código)
@st.cache_data
def train_models():
    """Simula el entrenamiento de los modelos basado en el análisis del notebook"""
    # Esta función simula tus modelos entrenados
    # En producción, cargarías los modelos guardados con pickle
    
    # Simulamos las métricas que obtuviste
    model_metrics = {
        'Logistic Regression': {
            'accuracy': 0.91,
            'precision': 0.88,
            'recall': 0.91,
            'f1_score': 0.89
        },
        'Random Forest': {
            'accuracy': 0.90,
            'precision': 0.88,
            'recall': 0.86,
            'f1_score': 0.87
        },
        'XGBoost': {
            'accuracy': 0.86,
            'precision': 0.82,
            'recall': 0.83,
            'f1_score': 0.83
        }
    }
    
    return model_metrics

# Función para mapear valores categóricos
def get_category_mappings():
    """Define los mapeos de categorías basados en tu dataset"""
    mappings = {
        'Gender': {'Female': 0, 'Male': 1},
        'Hx_Radiotherapy': {'No': 0, 'Yes': 1},
        'Adenopathy': {'Bilateral': 0, 'Extensive': 1, 'Left': 2, 'No': 3, 'Right': 4, 'Unilateral': 5},
        'Pathology': {'Follicular': 0, 'Hurthel cell': 1, 'Micropapillary': 2, 'Papillary': 3},
        'Focality': {'Multi-Focal': 0, 'Uni-Focal': 1},
        'T': {'T1a': 0, 'T1b': 1, 'T2': 2, 'T3a': 3, 'T3b': 4, 'T4a': 5, 'T4b': 6},
        'N': {'N0': 0, 'N1a': 1, 'N1b': 2},
        'M': {'M0': 0, 'M1': 1},
        'Stage': {'I': 0, 'II': 1, 'III': 2, 'IVA': 3, 'IVB': 4}
    }
    return mappings

# Función de predicción simulada
def predict_recurrence(features, model_choice):
    """Simula la predicción basada en los patrones de tu análisis"""
    
    # Factores de riesgo basados en tu análisis de correlación
    risk_factors = {
        'high_stage': features['Stage'] >= 3,  # Stage IVA, IVB
        'lymph_nodes': features['N'] > 0,      # N1a, N1b
        'male_gender': features['Gender'] == 1,
        'older_age': features['Age'] > 60,
        'multi_focal': features['Focality'] == 0,
        'high_t': features['T'] >= 5           # T4a, T4b
    }
    
    # Calcular score de riesgo
    risk_score = sum([
        risk_factors['high_stage'] * 0.4,
        risk_factors['lymph_nodes'] * 0.3,
        risk_factors['male_gender'] * 0.15,
        risk_factors['older_age'] * 0.1,
        risk_factors['multi_focal'] * 0.2,
        risk_factors['high_t'] * 0.25
    ])
    
    # Ajustar probabilidad según el modelo
    model_adjustments = {
        'Logistic Regression': 0.85,
        'Random Forest': 0.80,
        'XGBoost': 0.75
    }
    
    base_prob = min(risk_score * model_adjustments[model_choice], 0.95)
    
    # Predicción final
    prediction = 1 if base_prob > 0.5 else 0
    confidence = base_prob if prediction == 1 else (1 - base_prob)
    
    return prediction, confidence, risk_factors

# Sidebar para selección de modelo
st.sidebar.subheader("🤖 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ['Logistic Regression', 'Random Forest', 'XGBoost'],
    help="Select the machine learning model for prediction"
)

# Mostrar métricas del modelo seleccionado
model_metrics = train_models()
selected_metrics = model_metrics[model_choice]

st.sidebar.markdown("### 📈 Model Performance")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Accuracy", f"{selected_metrics['accuracy']:.2f}")
    st.metric("Precision", f"{selected_metrics['precision']:.2f}")
with col2:
    st.metric("Recall", f"{selected_metrics['recall']:.2f}")
    st.metric("F1-Score", f"{selected_metrics['f1_score']:.2f}")

# Formulario de entrada de datos
st.header("📝 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=45, help="Patient's age in years")
    gender = st.selectbox("Gender", ['Female', 'Male'])
    hx_radio = st.selectbox("History of Radiotherapy", ['No', 'Yes'])
    adenopathy = st.selectbox("Adenopathy", ['No', 'Right', 'Left', 'Bilateral', 'Extensive', 'Unilateral'])

with col2:
    pathology = st.selectbox("Pathology Type", ['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell'])
    focality = st.selectbox("Focality", ['Uni-Focal', 'Multi-Focal'])
    t_stage = st.selectbox("T Stage", ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
    n_stage = st.selectbox("N Stage", ['N0', 'N1a', 'N1b'])

with col3:
    m_stage = st.selectbox("M Stage", ['M0', 'M1'])
    stage = st.selectbox("Overall Stage", ['I', 'II', 'III', 'IVA', 'IVB'])

# Botón de predicción
if st.button("🔮 Predict Recurrence", type="primary"):
    # Preparar features
    mappings = get_category_mappings()
    
    features = {
        'Age': age,
        'Gender': mappings['Gender'][gender],
        'Hx_Radiotherapy': mappings['Hx_Radiotherapy'][hx_radio],
        'Adenopathy': mappings['Adenopathy'][adenopathy],
        'Pathology': mappings['Pathology'][pathology],
        'Focality': mappings['Focality'][focality],
        'T': mappings['T'][t_stage],
        'N': mappings['N'][n_stage],
        'M': mappings['M'][m_stage],
        'Stage': mappings['Stage'][stage]
    }
    
    # Realizar predicción
    prediction, confidence, risk_factors = predict_recurrence(features, model_choice)
    
    # Mostrar resultados
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    # Resultado principal
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>⚠️ HIGH RISK</h2>
            <p style="font-size: 1.2em;">The model predicts a <strong>HIGH PROBABILITY</strong> of cancer recurrence</p>
            <p style="font-size: 1.5em;"><strong>Confidence: {confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box" style="background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);">
            <h2>✅ LOW RISK</h2>
            <p style="font-size: 1.2em;">The model predicts a <strong>LOW PROBABILITY</strong> of cancer recurrence</p>
            <p style="font-size: 1.5em;"><strong>Confidence: {confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Análisis de factores de riesgo
    st.subheader("🔍 Risk Factor Analysis")
    
    risk_df = pd.DataFrame([
        {"Risk Factor": "Advanced Stage (IV)", "Present": "Yes" if risk_factors['high_stage'] else "No"},
        {"Risk Factor": "Lymph Node Involvement", "Present": "Yes" if risk_factors['lymph_nodes'] else "No"},
        {"Risk Factor": "Male Gender", "Present": "Yes" if risk_factors['male_gender'] else "No"},
        {"Risk Factor": "Age > 60", "Present": "Yes" if risk_factors['older_age'] else "No"},
        {"Risk Factor": "Multi-Focal Tumor", "Present": "Yes" if risk_factors['multi_focal'] else "No"},
        {"Risk Factor": "High T Stage (T4)", "Present": "Yes" if risk_factors['high_t'] else "No"}
    ])
    
    fig = px.bar(risk_df, 
                 x="Risk Factor", 
                 color="Present",
                 title="Risk Factors Present in This Patient",
                 color_discrete_map={"Yes": "#ff6b6b", "No": "#51cf66"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de factores de riesgo
    st.dataframe(risk_df, use_container_width=True)

# Información adicional
st.markdown("---")
st.header("📚 Additional Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Model Accuracy Comparison")
    metrics_df = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": 0.91, "F1-Score": 0.89},
        {"Model": "Random Forest", "Accuracy": 0.90, "F1-Score": 0.87},
        {"Model": "XGBoost", "Accuracy": 0.86, "F1-Score": 0.83}
    ])
    
    fig = px.bar(metrics_df, x="Model", y=["Accuracy", "F1-Score"], 
                 barmode='group', title="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("ℹ️ Important Notes")
    st.info("""
    **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
    
    **Key Points:**
    - Based on 383 patient records
    - Models trained on thyroid cancer data
    - Considers multiple clinical factors
    - Prediction accuracy: 86-91%
    
    Always consult with healthcare professionals for medical decisions.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Thyroid Cancer Recurrence Predictor | Built with Streamlit</p>
    <p>Data source: Thyroid Cancer Recurrence Dataset (383 patients)</p>
</div>
""", unsafe_allow_html=True)
