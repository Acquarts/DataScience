# Ferrari Image Recommendation System 🏎️

Un sistema de recomendación de imágenes inteligente para vehículos Ferrari utilizando Deep Learning y similitud coseno.

## 📋 Descripción

Este proyecto implementa un sistema de recomendación basado en contenido visual que utiliza la red neuronal convolucional ResNet50 pre-entrenada para extraer características de imágenes de Ferrari y recomendar vehículos similares basándose en similitud coseno.

## ✨ Características

- **🤖 Deep Learning**: Utiliza ResNet50 pre-entrenado en ImageNet
- **📊 Similitud Coseno**: Calcula similitudes entre vectores de características
- **🖼️ Visualización**: Muestra resultados de recomendaciones visualmente
- **⚡ Procesamiento Eficiente**: Manejo optimizado de datasets de imágenes
- **🎯 Alta Precisión**: Recomendaciones basadas en características visuales profundas

## 🛠️ Tecnologías

- **Python 3.x**
- **TensorFlow/Keras** - Para el modelo ResNet50
- **NumPy** - Operaciones numéricas
- **Pandas** - Manipulación de datos
- **Scikit-learn** - Similitud coseno
- **PIL (Pillow)** - Procesamiento de imágenes
- **Matplotlib** - Visualización
- **tqdm** - Barras de progreso

## 📦 Instalación

1. **Clona el repositorio**
```bash
git clone https://github.com/tu-usuario/ferrari-recommendation-system.git
cd ferrari-recommendation-system
```

2. **Instala las dependencias**
```bash
pip install tensorflow numpy pandas scikit-learn Pillow matplotlib tqdm
```

## 📁 Estructura del Proyecto

```
ferrari-recommendation-system/
│
├── ferrari_recommendation.py          # Código principal
├── ferrari_metadata.csv              # Metadatos del dataset
├── ferrari_dataset/                  # Dataset de imágenes
│   └── ferrari_images/
│       ├── 512/                     # Modelos Ferrari 512
│       ├── roma/                    # Ferrari Roma
│       ├── formula_1/               # Vehículos F1
│       └── ...
├── test_images/                      # Imágenes de prueba
│   ├── test_ferrari1.jpg
│   └── test_ferrari2.jpg
└── README.md
```

## 🚀 Uso

### 1. Configuración Inicial

```python
import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo ResNet50 pre-entrenado
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
```

### 2. Procesamiento del Dataset

```python
# Configurar rutas
catalog_dir = "path/to/ferrari-dataset"
csv_path = "path/to/ferrari_metadata.csv"

# Procesar imágenes del catálogo
df = pd.read_csv(csv_path)
features = []
image_names = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row['image_path']
    img_path = os.path.join(catalog_dir, img_name)
    img_array = load_and_preprocess_image(img_path)
    if img_array is not None:
        feat = model.predict(img_array)[0]
        features.append(feat)
        image_names.append(img_name)
```

### 3. Generar Recomendaciones

```python
# Función de recomendación
def recommend_similar_images(query_image_path, top_k=5):
    img_array = load_and_preprocess_image(query_image_path)
    query_vector = model.predict(img_array)[0].reshape(1, -1)
    similarity_scores = cosine_similarity(query_vector, np.array(features))[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {image_names[idx]} — similarity: {similarity_scores[idx]:.4f}")

# Ejemplo de uso
recommend_similar_images("path/to/query_image.jpg", top_k=5)
```

### 4. Visualización de Resultados

```python
# Mostrar recomendaciones visualmente
show_similar_images("path/to/query_image.jpg", top_k=5)
```

## 📊 Resultados Ejemplo

### Consulta: 1970 Ferrari 512
```
🔎 Query image: 1970_Ferrari_512_M_2.jpg

1. ferrari_dataset/ferrari_images/512/1970_Ferrari_512_M_2.jpg — similarity: 1.0000
2. ferrari_dataset/ferrari_images/512/1970_Ferrari_512_M_1.jpg — similarity: 0.8254
3. ferrari_dataset/ferrari_images/512/1970_Ferrari_512_M_3.jpg — similarity: 0.7996
4. ferrari_dataset/ferrari_images/512/1970_Ferrari_512_S_4.jpg — similarity: 0.7349
5. ferrari_dataset/ferrari_images/formula_1/2024_Ferrari_SF-24_2.jpg — similarity: 0.7319
```

### Consulta: 2024 Ferrari Roma Spider
```
🔎 Query image: 2024_Ferrari_Roma_Spider_1.jpg

1. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_1.jpg — similarity: 1.0000
2. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_3.jpg — similarity: 0.7566
3. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_2.jpg — similarity: 0.7505
4. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_6.jpg — similarity: 0.7405
5. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_4.jpg — similarity: 0.7358
```

## 🧠 Algoritmo

1. **Extracción de Características**: Utiliza ResNet50 para extraer vectores de características de 2048 dimensiones
2. **Similitud Coseno**: Calcula la similitud entre el vector consulta y todos los vectores del dataset
3. **Ranking**: Ordena los resultados por similitud descendente
4. **Top-K**: Retorna las K imágenes más similares

## ⚙️ Funciones Principales

- `load_and_preprocess_image()`: Carga y preprocesa imágenes para el modelo
- `recommend_similar_images()`: Genera recomendaciones de texto
- `show_similar_images()`: Visualiza recomendaciones con matplotlib
- `recommend_from_path()`: Wrapper para recomendaciones desde archivos externos

## 📈 Rendimiento

- **Procesamiento**: ~100ms por imagen en GPU
- **Precisión**: Alta similitud visual entre recomendaciones
- **Escalabilidad**: Eficiente para datasets de miles de imágenes
- **Memoria**: Vectores de características compactos (2048 dim)

## 🔧 Configuración Avanzada

### Ajustar Parámetros del Modelo
```python
# Cambiar tamaño de entrada
target_size = (224, 224)  # Tamaño estándar ResNet50

# Ajustar número de recomendaciones
top_k = 10  # Incrementar para más resultados
```

### Optimización de GPU
```python
# Configurar GPU (si está disponible)
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **ImageNet** - Dataset de pre-entrenamiento
- **TensorFlow/Keras** - Framework de Deep Learning
- **Ferrari** - Por crear vehículos tan icónicos
- **ResNet** - Arquitectura de red neuronal

## 📧 Contacto

Adrián Zambrana - [Linkedin](https://www.linkedin.com/in/adrianzambranaacquaroni/) - info.aza.future@gmail.com

Link del Proyecto: [https://github.com/tu-usuario/ferrari-recommendation-system](https://github.com/tu-usuario/ferrari-recommendation-system)

---


