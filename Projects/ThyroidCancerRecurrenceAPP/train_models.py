"""
Thyroid Cancer Recurrence - Model Training Script
=================================================
Este script entrena los modelos de ML basado en el análisis del notebook
y guarda los modelos entrenados como archivos .pkl para usar en Streamlit
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(csv_path='filtered_thyroid_data.csv'):
    """
    Carga y preprocesa los datos exactamente como en el notebook
    """
    print("📊 Cargando datos...")
    df = pd.read_csv(csv_path)
    print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Guardar mapeos originales antes de LabelEncoder
    categorical_mappings = {}
    
    # Crear LabelEncoders para cada columna categórica
    label_encoders = {}
    
    print("🔄 Aplicando Label Encoding...")
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        
        # Guardar mapeo original
        unique_values = df[col].unique()
        categorical_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
        
        # Aplicar LabelEncoder
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
        print(f"   - {col}: {len(unique_values)} categorías únicas")
    
    return df, categorical_mappings, label_encoders

def prepare_features_and_target(df):
    """
    Prepara las características y variable objetivo como en el notebook
    """
    print("🎯 Preparando características y variable objetivo...")
    
    # Definir características y objetivo (exactamente como en tu notebook)
    X = df.drop(['Recurred', 'Risk', 'Response'], axis=1)
    y = df['Recurred']
    
    print(f"   - Características: {X.shape[1]} variables")
    print(f"   - Características usadas: {list(X.columns)}")
    print(f"   - Variable objetivo: {y.name}")
    print(f"   - Distribución objetivo: {y.value_counts().to_dict()}")
    
    return X, y

def train_and_evaluate_models(X, y, test_size=0.2, random_state=42):
    """
    Entrena y evalúa los modelos exactamente como en el notebook
    """
    print("🤖 Entrenando modelos...")
    
    # Escalado de características
    print("📏 Aplicando StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # División train/test (exactamente como en tu notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"   - Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"   - Datos de prueba: {X_test.shape[0]} muestras")
    
    # Diccionario para almacenar modelos y métricas
    models = {}
    metrics = {}
    
    # 1. Logistic Regression
    print("\n🔵 Entrenando Logistic Regression...")
    lr = LogisticRegression(random_state=random_state)
    lr.fit(X_train, y_train)
    predict_lr = lr.predict(X_test)
    
    report_lr = classification_report(y_test, predict_lr, output_dict=True)
    acc_lr = accuracy_score(y_test, predict_lr)
    
    models['Logistic Regression'] = lr
    metrics['Logistic Regression'] = {
        'accuracy': acc_lr,
        'precision': report_lr['macro avg']['precision'],
        'recall': report_lr['macro avg']['recall'],
        'f1_score': report_lr['macro avg']['f1-score'],
        'classification_report': report_lr
    }
    
    print("   ✅ Logistic Regression entrenado")
    print(f"      - Accuracy: {acc_lr:.3f}")
    print(f"      - F1-Score: {report_lr['macro avg']['f1-score']:.3f}")
    
    # 2. Random Forest
    print("\n🌳 Entrenando Random Forest...")
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, y_train)
    predict_rf = rf.predict(X_test)
    
    report_rf = classification_report(y_test, predict_rf, output_dict=True)
    acc_rf = accuracy_score(y_test, predict_rf)
    
    models['Random Forest'] = rf
    metrics['Random Forest'] = {
        'accuracy': acc_rf,
        'precision': report_rf['macro avg']['precision'],
        'recall': report_rf['macro avg']['recall'],
        'f1_score': report_rf['macro avg']['f1-score'],
        'classification_report': report_rf
    }
    
    print("   ✅ Random Forest entrenado")
    print(f"      - Accuracy: {acc_rf:.3f}")
    print(f"      - F1-Score: {report_rf['macro avg']['f1-score']:.3f}")
    
    # 3. XGBoost
    print("\n🚀 Entrenando XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    xgb.fit(X_train, y_train)
    predict_xgb = xgb.predict(X_test)
    
    report_xgb = classification_report(y_test, predict_xgb, output_dict=True)
    acc_xgb = accuracy_score(y_test, predict_xgb)
    
    models['XGBoost'] = xgb
    metrics['XGBoost'] = {
        'accuracy': acc_xgb,
        'precision': report_xgb['macro avg']['precision'],
        'recall': report_xgb['macro avg']['recall'],
        'f1_score': report_xgb['macro avg']['f1-score'],
        'classification_report': report_xgb
    }
    
    print("   ✅ XGBoost entrenado")
    print(f"      - Accuracy: {acc_xgb:.3f}")
    print(f"      - F1-Score: {report_xgb['macro avg']['f1-score']:.3f}")
    
    return models, metrics, scaler, X.columns.tolist()

def save_models_and_metadata(models, metrics, scaler, feature_names, categorical_mappings, label_encoders):
    """
    Guarda todos los modelos y metadatos necesarios
    """
    print("\n💾 Guardando modelos y metadatos...")
    
    # Guardar modelos individuales
    for model_name, model in models.items():
        filename = model_name.lower().replace(' ', '_') + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✅ {filename} guardado")
    
    # Guardar scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("   ✅ scaler.pkl guardado")
    
    # Guardar label encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("   ✅ label_encoders.pkl guardado")
    
    # Guardar métricas como JSON
    with open('model_metrics.json', 'w') as f:
        # Convertir numpy types a python types para JSON
        metrics_json = {}
        for model_name, model_metrics in metrics.items():
            metrics_json[model_name] = {
                'accuracy': float(model_metrics['accuracy']),
                'precision': float(model_metrics['precision']),
                'recall': float(model_metrics['recall']),
                'f1_score': float(model_metrics['f1_score'])
            }
        json.dump(metrics_json, f, indent=4)
    print("   ✅ model_metrics.json guardado")
    
    # Guardar mapeos categóricos
    with open('categorical_mappings.json', 'w') as f:
        json.dump(categorical_mappings, f, indent=4)
    print("   ✅ categorical_mappings.json guardado")
    
    # Guardar nombres de características
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=4)
    print("   ✅ feature_names.json guardado")

def print_summary(metrics):
    """
    Imprime resumen final de los modelos
    """
    print("\n" + "="*60)
    print("📈 RESUMEN FINAL DE MODELOS")
    print("="*60)
    
    print(f"{'Modelo':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    
    for model_name, model_metrics in metrics.items():
        print(f"{model_name:<20} {model_metrics['accuracy']:<10.3f} "
              f"{model_metrics['precision']:<10.3f} {model_metrics['recall']:<10.3f} "
              f"{model_metrics['f1_score']:<10.3f}")
    
    # Encontrar el mejor modelo
    best_model = max(metrics.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Mejor modelo: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS THYROID CANCER RECURRENCE")
    print("="*70)
    
    try:
        # 1. Cargar y preprocesar datos
        df, categorical_mappings, label_encoders = load_and_preprocess_data()
        
        # 2. Preparar características y objetivo
        X, y = prepare_features_and_target(df)
        
        # 3. Entrenar modelos
        models, metrics, scaler, feature_names = train_and_evaluate_models(X, y)
        
        # 4. Guardar todo
        save_models_and_metadata(models, metrics, scaler, feature_names, 
                                categorical_mappings, label_encoders)
        
        # 5. Mostrar resumen
        print_summary(metrics)
        
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("🎯 Los modelos están listos para usar en Streamlit")
        print("📁 Archivos generados:")
        print("   - logistic_regression.pkl")
        print("   - random_forest.pkl") 
        print("   - xgboost.pkl")
        print("   - scaler.pkl")
        print("   - label_encoders.pkl")
        print("   - model_metrics.json")
        print("   - categorical_mappings.json")
        print("   - feature_names.json")
        print("="*70)
        
    except Exception as e:
        print(f"❌ ERROR durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()
