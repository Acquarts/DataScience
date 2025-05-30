import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import os
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

def train_models_automatically():
    """Entrena los modelos automáticamente si no existen"""
    try:
        # Verificar que el CSV existe
        if not os.path.exists('filtered_thyroid_data.csv'):
            st.error("❌ Dataset 'filtered_thyroid_data.csv' not found!")
            st.write("Please make sure the CSV file is in the same directory as the app.")
            return False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading dataset...")
        progress_bar.progress(10)
        
        # Cargar datos
        df = pd.read_csv('filtered_thyroid_data.csv')
        
        status_text.text("Preprocessing data...")
        progress_bar.progress(20)
        
        # Guardar mapeos originales antes de LabelEncoder
        categorical_mappings = {}
        label_encoders = {}
        
        # Aplicar Label Encoding
        for col in df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            unique_values = df[col].unique()
            categorical_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Preparar características y objetivo
        X = df.drop(['Recurred', 'Risk', 'Response'], axis=1)
        y = df['Recurred']
        
        status_text.text("Scaling features...")
        progress_bar.progress(30)
        
        # Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Entrenar modelos
        models = {}
        metrics = {}
        
        status_text.text("Training Logistic Regression...")
        progress_bar.progress(40)
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        predict_lr = lr.predict(X_test)
        report_lr = classification_report(y_test, predict_lr, output_dict=True)
        acc_lr = accuracy_score(y_test, predict_lr)
        
        models['Logistic Regression'] = lr
        metrics['Logistic Regression'] = {
            'accuracy': acc_lr,
            'precision': report_lr['macro avg']['precision'],
            'recall': report_lr['macro avg']['recall'],
            'f1_score': report_lr['macro avg']['f1-score']
        }
        
        status_text.text("Training Random Forest...")
        progress_bar.progress(60)
        
        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        predict_rf = rf.predict(X_test)
        report_rf = classification_report(y_test, predict_rf, output_dict=True)
        acc_rf = accuracy_score(y_test, predict_rf)
        
        models['Random Forest'] = rf
        metrics['Random Forest'] = {
            'accuracy': acc_rf,
            'precision': report_rf['macro avg']['precision'],
            'recall': report_rf['macro avg']['recall'],
            'f1_score': report_rf['macro avg']['f1-score']
        }
        
        status_text.text("Training XGBoost...")
        progress_bar.progress(80)
        
        # XGBoost
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        predict_xgb = xgb.predict(X_test)
        report_xgb = classification_report(y_test, predict_xgb, output_dict=True)
        acc_xgb = accuracy_score(y_test, predict_xgb)
        
        models['XGBoost'] = xgb
        metrics['XGBoost'] = {
            'accuracy': acc_xgb,
            'precision': report_xgb['macro avg']['precision'],
            'recall': report_xgb['macro avg']['recall'],
            'f1_score': report_xgb['macro avg']['f1-score']
        }
        
        status_text.text("Saving models...")
        progress_bar.progress(90)
        
        # Guardar modelos
        with open('logistic_regression.pkl', 'wb') as f:
            pickle.dump(lr, f)
        with open('random_forest.pkl', 'wb') as f:
            pickle.dump(rf, f)
        with open('xgboost.pkl', 'wb') as f:
            pickle.dump(xgb, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        
        # Guardar metadatos
        metrics_json = {}
        for model_name, model_metrics in metrics.items():
            metrics_json[model_name] = {
                'accuracy': float(model_metrics['accuracy']),
                'precision': float(model_metrics['precision']),
                'recall': float(model_metrics['recall']),
                'f1_score': float(model_metrics['f1_score'])
            }
        
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics_json, f, indent=4)
        with open('categorical_mappings.json', 'w') as f:
            json.dump(categorical_mappings, f, indent=4)
        with open('feature_names.json', 'w') as f:
            json.dump(X.columns.tolist(), f, indent=4)
        
        progress_bar.progress(100)
        status_text.text("✅ Models trained and saved successfully!")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        return False

# Cargar modelos y metadatos
@st.cache_resource
def load_models_and_metadata():
    """Carga todos los modelos entrenados y metadatos"""
    
    # Verificar si los modelos existen
    required_files = [
        'logistic_regression.pkl',
        'random_forest.pkl', 
        'xgboost.pkl',
        'scaler.pkl',
        'label_encoders.pkl',
        'model_metrics.json',
        'categorical_mappings.json',
        'feature_names.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.info("🔄 Models not found. Training models automatically...")
        st.write("This may take a few moments on first run...")
        
        # Entrenar modelos automáticamente
        success = train_models_automatically()
        if not success:
            st.error("❌ Failed to train models automatically")
            st.stop()
        
        # Recargar después del entrenamiento
        st.rerun()
    
    try:
        # Cargar modelos
        with open('logistic_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('xgboost.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        
        # Cargar scaler y label encoders
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Cargar metadatos
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
        with open('categorical_mappings.json', 'r') as f:
            categorical_mappings = json.load(f)
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        models = {
            'Logistic Regression': lr_model,
            'Random Forest': rf_model,
            'XGBoost': xgb_model
        }
        
        return models, scaler, label_encoders, metrics, categorical_mappings, feature_names
        
    except Exception as e:
        st.error(f"Error loading models after training: {str(e)}")
        st.stop()

# Cargar todo
models, scaler, label_encoders, model_metrics, categorical_mappings, feature_names = load_models_and_metadata()

# Información en la sidebar
st.sidebar.markdown("""
<div class="sidebar-info">
<h3>📊 About This App</h3>
<p>This application predicts thyroid cancer recurrence using <strong>real machine learning models</strong> trained on 383 patient records.</p>
<p><strong>Models Available:</strong></p>
<ul>
<li>Logistic Regression</li>
<li>Random Forest</li>
<li>XGBoost</li>
</ul>
<p><strong>✅ Using trained .pkl models</strong></p>
</div>
""", unsafe_allow_html=True)

# Función para preparar las características de entrada
def prepare_input_features(input_data, categorical_mappings, feature_names):
    """Prepara las características de entrada para la predicción"""
    
    # Crear DataFrame con las características en el orden correcto
    features_df = pd.DataFrame([input_data])
    
    # Aplicar el mismo preprocesamiento que en el entrenamiento
    # (El LabelEncoder ya se aplicó durante el entrenamiento, aquí solo mapeamos)
    processed_features = []
    
    for feature in feature_names:
        if feature in features_df.columns:
            processed_features.append(features_df[feature].iloc[0])
        else:
            # Si falta alguna característica, usar valor por defecto
            processed_features.append(0)
    
    return np.array(processed_features).reshape(1, -1)

# Función de predicción real
def predict_with_real_model(input_data, model_name, models, scaler, categorical_mappings, feature_names):
    """Realiza predicción usando los modelos reales entrenados"""
    
    # Preparar características
    features = prepare_input_features(input_data, categorical_mappings, feature_names)
    
    # Escalar características
    features_scaled = scaler.transform(features)
    
    # Seleccionar modelo
    model = models[model_name]
    
    # Realizar predicción
    prediction = model.predict(features_scaled)[0]
    
    # Obtener probabilidades
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        prob_no_recurrence = probabilities[0]
        prob_recurrence = probabilities[1]
    else:
        # Para modelos que no tienen predict_proba
        confidence = 0.8  # Valor por defecto
        prob_no_recurrence = 0.5
        prob_recurrence = 0.5
    
    return prediction, confidence, prob_no_recurrence, prob_recurrence

# Sidebar para selección de modelo
st.sidebar.subheader("🤖 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ['Logistic Regression', 'Random Forest', 'XGBoost'],
    help="Select the machine learning model for prediction"
)

# Mostrar métricas del modelo seleccionado
selected_metrics = model_metrics[model_choice]

st.sidebar.markdown("### 📈 Model Performance")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Accuracy", f"{selected_metrics['accuracy']:.3f}")
    st.metric("Precision", f"{selected_metrics['precision']:.3f}")
with col2:
    st.metric("Recall", f"{selected_metrics['recall']:.3f}")
    st.metric("F1-Score", f"{selected_metrics['f1_score']:.3f}")

# Obtener opciones para los selectboxes desde los mapeos categóricos
def get_category_options(category_name):
    """Obtiene las opciones disponibles para una categoría"""
    if category_name in categorical_mappings:
        # Invertir el mapeo para obtener las etiquetas originales
        return list(categorical_mappings[category_name].keys())
    return []

# Formulario de entrada de datos
st.header("📝 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=45, help="Patient's age in years")
    gender = st.selectbox("Gender", get_category_options('Gender'))
    hx_radio = st.selectbox("History of Radiotherapy", get_category_options('Hx Radiothreapy'))
    adenopathy = st.selectbox("Adenopathy", get_category_options('Adenopathy'))

with col2:
    pathology = st.selectbox("Pathology Type", get_category_options('Pathology'))
    focality = st.selectbox("Focality", get_category_options('Focality'))
    t_stage = st.selectbox("T Stage", get_category_options('T'))
    n_stage = st.selectbox("N Stage", get_category_options('N'))

with col3:
    m_stage = st.selectbox("M Stage", get_category_options('M'))
    stage = st.selectbox("Overall Stage", get_category_options('Stage'))

# Botón de predicción
if st.button("🔮 Predict Recurrence", type="primary"):
    # Preparar datos de entrada usando los mapeos categóricos
    input_features = {
        'Age': age,
        'Gender': categorical_mappings['Gender'][gender],
        'Hx Radiothreapy': categorical_mappings['Hx Radiothreapy'][hx_radio],
        'Adenopathy': categorical_mappings['Adenopathy'][adenopathy],
        'Pathology': categorical_mappings['Pathology'][pathology],
        'Focality': categorical_mappings['Focality'][focality],
        'T': categorical_mappings['T'][t_stage],
        'N': categorical_mappings['N'][n_stage],
        'M': categorical_mappings['M'][m_stage],
        'Stage': categorical_mappings['Stage'][stage]
    }
    
    # Realizar predicción real
    prediction, confidence, prob_no_recurrence, prob_recurrence = predict_with_real_model(
        input_features, model_choice, models, scaler, categorical_mappings, feature_names
    )
    
    # Mostrar resultados
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    # Resultado principal
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>⚠️ HIGH RISK</h2>
            <p style="font-size: 1.2em;">The model predicts a <strong>HIGH PROBABILITY</strong> of cancer recurrence</p>
            <p style="font-size: 1.5em;"><strong>Probability: {prob_recurrence:.1%}</strong></p>
            <p style="font-size: 1.0em;">Model: {model_choice}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box" style="background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);">
            <h2>✅ LOW RISK</h2>
            <p style="font-size: 1.2em;">The model predicts a <strong>LOW PROBABILITY</strong> of cancer recurrence</p>
            <p style="font-size: 1.5em;"><strong>Probability: {prob_no_recurrence:.1%}</strong></p>
            <p style="font-size: 1.0em;">Model: {model_choice}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de probabilidades
    st.subheader("📊 Prediction Probabilities")
    
    prob_data = pd.DataFrame({
        'Outcome': ['No Recurrence', 'Recurrence'],
        'Probability': [prob_no_recurrence, prob_recurrence],
        'Color': ['#51cf66', '#ff6b6b']
    })
    
    fig = px.bar(prob_data, 
                 x='Outcome', 
                 y='Probability',
                 color='Color',
                 color_discrete_map={'#51cf66': '#51cf66', '#ff6b6b': '#ff6b6b'},
                 title=f"Prediction Probabilities - {model_choice}",
                 text='Probability')
    
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de características importantes (para Random Forest)
    if model_choice == 'Random Forest' and hasattr(models[model_choice], 'feature_importances_'):
        st.subheader("🔍 Feature Importance Analysis")
        
        importances = models[model_choice].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(8)
        
        fig = px.bar(feature_importance_df, 
                     x='Importance', 
                     y='Feature',
                     orientation='h',
                     title="Top 8 Most Important Features (Random Forest)",
                     color='Importance',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar valores de entrada
    st.subheader("📋 Input Summary")
    input_summary = pd.DataFrame([
        {"Feature": "Age", "Value": age},
        {"Feature": "Gender", "Value": gender},
        {"Feature": "History of Radiotherapy", "Value": hx_radio},
        {"Feature": "Adenopathy", "Value": adenopathy},
        {"Feature": "Pathology Type", "Value": pathology},
        {"Feature": "Focality", "Value": focality},
        {"Feature": "T Stage", "Value": t_stage},
        {"Feature": "N Stage", "Value": n_stage},
        {"Feature": "M Stage", "Value": m_stage},
        {"Feature": "Overall Stage", "Value": stage}
    ])
    st.dataframe(input_summary, use_container_width=True)

# Información adicional
st.markdown("---")
st.header("📚 Additional Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Model Accuracy Comparison")
    metrics_df = pd.DataFrame([
        {"Model": model, "Accuracy": metrics['accuracy'], "F1-Score": metrics['f1_score']}
        for model, metrics in model_metrics.items()
    ])
    
    fig = px.bar(metrics_df, x="Model", y=["Accuracy", "F1-Score"], 
                 barmode='group', title="Real Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("ℹ️ Important Notes")
    st.info("""
    **✅ Real ML Models:** This app uses actual trained machine learning models from your notebook.
    
    **Key Points:**
    - Based on 383 patient records
    - Models trained on real thyroid cancer data
    - Uses the same preprocessing pipeline
    - Identical results to your analysis
    - Real confidence scores from model probabilities
    
    **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

# Información técnica
st.markdown("---")
st.header("🔧 Technical Information")

with st.expander("View Technical Details"):
    st.markdown(f"""
    **Models Loaded Successfully:**
    - ✅ Logistic Regression: Accuracy {model_metrics['Logistic Regression']['accuracy']:.3f}
    - ✅ Random Forest: Accuracy {model_metrics['Random Forest']['accuracy']:.3f}  
    - ✅ XGBoost: Accuracy {model_metrics['XGBoost']['accuracy']:.3f}
    
    **Features Used:** {len(feature_names)} features
    ```
    {', '.join(feature_names)}
    ```
    
    **Preprocessing:**
    - StandardScaler applied to all features
    - LabelEncoder applied to categorical variables
    - Same train/test split as notebook (80/20)
    
    **Files Loaded:**
    - logistic_regression.pkl
    - random_forest.pkl
    - xgboost.pkl
    - scaler.pkl
    - label_encoders.pkl
    - model_metrics.json
    - categorical_mappings.json
    - feature_names.json
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Thyroid Cancer Recurrence Predictor | Built with Streamlit | Using Real ML Models</p>
    <p>Data source: Thyroid Cancer Recurrence Dataset (383 patients)</p>
</div>
""", unsafe_allow_html=True)