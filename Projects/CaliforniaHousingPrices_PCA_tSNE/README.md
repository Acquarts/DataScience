# 🏠 Reducción de Dimensionalidad y Clustering en California Housing Dataset

## 🎯 Objetivo
Explorar la estructura de los datos del **California Housing Dataset** mediante **reducción de dimensionalidad** y **clustering no supervisado**, evaluando si es posible identificar patrones similares a la variable categórica `ocean_proximity` sin usarla directamente en el entrenamiento.

---

## 📂 Dataset
- **Fuente**: California Housing Dataset.
- **Filas**: 20.640
- **Columnas**: 9 (numéricas + `ocean_proximity` categórica)
- **Variable categórica objetivo (solo para evaluación)**: `ocean_proximity`.

---

## 🛠 Metodología

### 1. Preprocesamiento
- Codificación de la variable categórica (`OneHotEncoder`).
- Escalado de variables numéricas (`StandardScaler`).
- Pipeline unificado con `ColumnTransformer`.

---

### 2. PCA (Análisis de Componentes Principales)
- Reducción a **2 componentes**.
- **Varianza explicada acumulada**: ~62%.
- Visualización coloreando por `ocean_proximity` → alto solapamiento.
- Clustering con **K-Means (K=5)** → **ARI = 0.125** → baja similitud con categorías reales.

---

### 3. t-SNE (Reducción de Dimensionalidad No Lineal)
- Reducción a 2 dimensiones preservando relaciones no lineales.
- Parametrización inicial (`perplexity=30`, `learning_rate=200`) mostró mayor separación visual.
- Optimización de hiperparámetros:
  - Mejor combinación: **perplexity=30, learning_rate=500**.
  - **ARI con K-Means sobre t-SNE optimizado = 0.419** → mejora significativa frente a PCA.

---

### 4. Clustering (K-Means)
- **Número de clusters (K)**: igual al nº de categorías reales (5).
- Evaluación con **Adjusted Rand Index (ARI)** usando `ocean_proximity` como referencia.
- Tabla de contingencia → varios clusters representan claramente categorías específicas (ej. `INLAND`, `NEAR BAY`).

---

## 📊 Resultados

| Técnica                      | ARI   | Observaciones |
|------------------------------|-------|---------------|
| PCA + K-Means                | 0.125 | Mucho solapamiento, estructura lineal insuficiente. |
| t-SNE inicial + K-Means      | 0.341 | Mejora notable, grupos más definidos visualmente. |
| t-SNE optimizado + K-Means   | 0.419 | Mejor separación, varios clusters alineados con categorías reales. |

---

## 📈 Visualización Comparativa
- **Izquierda**: t-SNE optimizado coloreado por categorías reales.  
- **Derecha**: t-SNE optimizado coloreado por clusters K-Means.  
*(Inserta aquí las imágenes generadas en el notebook)*

---

## 🧠 Conclusiones
- PCA es útil para visualización y reducción rápida, pero limitado a patrones lineales.
- t-SNE capta relaciones no lineales, logrando mejor separación en este dataset.
- Aun sin usar `ocean_proximity` para entrenar, el clustering detectó parte de su estructura.
- ARI = 0.419 → correlación moderada entre clusters y categorías reales.

---

## 🚀 Próximos pasos
- Probar **UMAP** como alternativa más rápida y escalable a t-SNE.
- Enriquecer las features con transformaciones geográficas para mejorar aún más el clustering.
- Usar métodos de clustering más flexibles (Gaussian Mixtures, DBSCAN optimizado).

---

## ⚙️ Requisitos de ejecución
Instalar dependencias necesarias:
```bash
pip install -r requirements.txt
