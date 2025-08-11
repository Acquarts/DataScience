# Detección de Tumores Cerebrales con YOLOv8

## Descripción del Proyecto

Este proyecto implementa un sistema de detección de tumores cerebrales utilizando YOLOv8 (You Only Look Once versión 8) para identificar automáticamente la presencia de tumores en imágenes de resonancia magnética cerebral.

## Dataset

- **Nombre**: Brain Tumor Dataset
- **Clases**: 
  - `0: negative` - Sin tumor
  - `1: positive` - Con tumor
- **División**:
  - **Entrenamiento**: 893 imágenes
  - **Validación**: 223 imágenes
- **Formato**: Detección de objetos con bounding boxes

## Arquitectura del Modelo

- **Modelo base**: YOLOv8n (nano) - Versión ligera y rápida
- **Parámetros**: 3,011,238 parámetros entrenables
- **GFLOPs**: 8.2
- **Capas**: 129 capas en total

## Configuración de Entrenamiento

```python
model = YOLO('yolov8n.pt')
model.train(
    data='datasets/brain-tumor.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device=0  # GPU
)
```

### Hiperparámetros

- **Épocas**: 50
- **Tamaño de imagen**: 640x640 píxeles
- **Batch size**: 8
- **Optimizador**: AdamW (lr=0.001667, momentum=0.9)
- **Data augmentation**: Activado (mosaic, mixup, albumentations)

## Resultados del Entrenamiento

### Métricas Finales
- **mAP@50**: 49.9% - Precisión media a IoU 0.5
- **mAP@50-95**: 36.7% - Precisión media en rango IoU 0.5-0.95
- **Precision**: 42.0% - Proporción de detecciones correctas
- **Recall**: 82.0% - Proporción de tumores detectados

### Métricas por Clase
| Clase | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| Negative | 54.5% | 70.8% | 59.1% | 43.7% |
| Positive | 29.6% | 93.1% | 40.8% | 29.7% |

### Rendimiento
- **Velocidad de inferencia**: ~1ms por imagen
- **Tiempo de entrenamiento**: 0.130 horas (7.8 minutos)

## Análisis de Resultados

### Fortalezas
- **Alto Recall (82%)**: El modelo detecta la mayoría de tumores presentes
- **Velocidad**: Inferencia muy rápida, adecuada para aplicaciones en tiempo real
- **Eficiencia**: Modelo ligero con pocos parámetros

### Áreas de Mejora
- **Precisión moderada (42%)**: Genera algunos falsos positivos
- **Desbalance entre clases**: La clase "positive" tiene menor precisión
- **Dataset pequeño**: Solo 893 imágenes de entrenamiento

## Instalación y Uso

### Requisitos
```bash
pip install ultralytics torch torchvision matplotlib
```

### Entrenamiento
```python
from ultralytics import YOLO

# Cargar modelo preentrenado
model = YOLO('yolov8n.pt')

# Entrenar
results = model.train(
    data='path/to/brain-tumor.yaml',
    epochs=50,
    imgsz=640,
    batch=8
)
```

### Inferencia
```python
# Cargar modelo entrenado
model = YOLO('runs/detect/train/weights/best.pt')

# Predecir en nueva imagen
results = model.predict('path/to/image.jpg', conf=0.25)

# Visualizar resultados
import matplotlib.pyplot as plt
plt.imshow(results[0].plot())
plt.show()
```

## Estructura del Proyecto

```
brain-tumor-detection/
├── datasets/
│   └── brain-tumor/
│       ├── train/
│       ├── valid/
│       └── brain-tumor.yaml
├── runs/
│   └── detect/
│       └── train/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── results.png
└── README.md
```

## Aplicaciones Potenciales

- **Screening médico**: Detección temprana de tumores cerebrales
- **Asistencia diagnóstica**: Apoyo a radiólogos en la interpretación de RM
- **Investigación**: Análisis automatizado de grandes datasets médicos
- **Telemedicina**: Diagnóstico remoto en áreas con recursos limitados

## Consideraciones Éticas y Médicas

⚠️ **Importante**: Este modelo es únicamente para fines educativos y de investigación. No debe utilizarse como herramienta de diagnóstico médico sin supervisión profesional adecuada.

## Trabajo Futuro

1. **Aumento del dataset**: Incluir más imágenes y diversidad de casos
2. **Mejora de arquitectura**: Probar YOLOv8s o YOLOv8m para mayor precisión
3. **Segmentación**: Implementar segmentación semántica del tumor
4. **Validación clínica**: Evaluación con especialistas médicos
5. **Optimización**: Cuantización y pruning para deployment móvil

## Referencias

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [Brain Tumor Dataset](https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip)

## Licencia

Este proyecto utiliza licencia AGPL-3.0 de Ultralytics YOLO.