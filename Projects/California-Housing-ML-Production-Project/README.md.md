# 🏠 California Housing — ML Pipeline, API y Frontend

Proyecto completo de **Machine Learning** para predecir el valor medio de la vivienda (`median_house_value`) en distritos de California. Incluye **análisis exploratorio (EDA)**, **entrenamiento comparativo de modelos**, **API REST** con **FastAPI** y **frontend** en **Streamlit** para realizar predicciones individuales y por lotes (CSV).

---

## 📌 Funcionalidades principales

- **EDA**: análisis descriptivo, distribución de variables, detección de nulos, correlaciones.
- **Preprocesamiento**: imputación de valores, escalado de numéricas y codificación one-hot para categóricas.
- **Modelos probados**:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `XGBRegressor`
- Guarda automáticamente el **mejor modelo por RMSE** en `artifacts/model.joblib`.
- **API REST**: endpoint `/predict` para recibir datos y devolver el precio estimado.
- **Frontend**:
  - Predicción **individual** vía formulario (API).
  - Predicción **por lotes** desde un archivo CSV procesado localmente.
- **Tests** de la API y del pipeline con `pytest`.

---

## 📂 Estructura del proyecto

```
california-housing-ml/
├── .streamlit/
│   └── secrets.toml          # Config opcional (API_URL para Streamlit)
├── api/
│   └── app.py               # API FastAPI con /predict y /health
├── app/
│   └── streamlit_app.py     # App Streamlit (individual + CSV)
├── artifacts/
│   └── model.joblib         # Pipeline entrenado (preprocesado + modelo)
├── data/
│   └── housing.csv          # Dataset original (opcional)
├── eda/
│   └── eda_california_housing.ipynb  # Análisis exploratorio
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuración general
│   ├── data.py              # Funciones de carga de datos
│   ├── evaluate.py          # Evaluación del modelo
│   ├── pipeline.py          # Definición de transformaciones
│   ├── predict.py           # Predicción local
│   ├── train.py             # Entrenamiento de modelos
│   └── utils.py             # Funciones auxiliares
├── tests/
│   ├── test_api.py          # Test de la API
│   └── test_pipeline.py     # Test del pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu_usuario/california-housing-ml.git
cd california-housing-ml

# 2. Crear entorno virtual
python -m venv .venv
# Activar (Linux/Mac)
source .venv/bin/activate
# Activar (Windows)
.venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

💡 **Si hay problemas con xgboost en Windows:**

```bash
conda install -c conda-forge xgboost
```

## 🧪 Entrenamiento y evaluación

```bash
# Entrenar y guardar el mejor modelo en artifacts/model.joblib
python -m src.train

# Evaluar el modelo entrenado
python -m src.evaluate
```

**Ejemplo de resultados (pueden variar):**

| Modelo | MAE | RMSE | R² |
|--------|-----|------|-----|
| LinearRegression | 50670.49 | 70059.19 | 0.625 |
| RandomForest | 31393.36 | 48676.22 | 0.819 |
| XGBRegressor | 30235.84 | 45930.94 | 0.839 |

## 🚀 API con FastAPI

```bash
uvicorn api.app:app --reload
```

**Documentación interactiva:** http://127.0.0.1:8000/docs

### Endpoints:

- **GET** `/health` → comprueba estado
- **POST** `/predict` → predice precio de una vivienda

**Ejemplo JSON:**

```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```

**Respuesta:**

```json
{ "predicted_price": 426046.59 }
```

## 🖥️ Frontend con Streamlit

Lanza un formulario para predicción individual y subida de CSV para predicción por lotes.

```bash
streamlit run app/streamlit_app.py
```

### Config opcional para producción

`.streamlit/secrets.toml`:

```toml
API_URL = "https://mi-api.com/predict"
```

### Formato CSV para predicción por lotes:

```csv
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity
```

💡 **Recuerda que** `total_rooms`, `total_bedrooms`, `population` y `households` son agregados a nivel de distrito.

## 📊 EDA

En `eda/eda_california_housing.ipynb` encontrarás:

- Información general (`df.info()`, `df.describe()`).
- Distribución de variables numéricas y categóricas.
- Mapa de calor de correlaciones.
- Relación geográfica entre ubicación (latitude / longitude) y precio.

## 🧰 Tests

```bash
pytest -q
```

- `test_pipeline.py`: comprueba que el pipeline se entrena y predice correctamente.
- `test_api.py`: testea los endpoints `/health` y `/predict`.

## 📦 requirements.txt

```txt
pandas
numpy
scikit-learn
xgboost
fastapi
uvicorn
pydantic
joblib
matplotlib
seaborn
streamlit
requests
ipykernel
pytest
```

## 🧹 .gitignore recomendado

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
.DS_Store

# Entornos
.venv/
venv/
.env
.env.*

# Streamlit
.streamlit/secrets.toml

# Datos y artefactos
# artifacts/
# data/
```