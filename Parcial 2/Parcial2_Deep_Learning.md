> **PARCIAL** **2** **–** **DEEP** **LEARNING**

**Objetivo**

Desarrollar un clasificador de imágenes industriales comparando tres
estrategias de modelado, aplicando los principios de diseño y
entrenamiento de redes neuronales vistos en el libro de Andrew Ng.

**Dataset**

[Industrial Classification Data Set
(Kaggle)](https://www.kaggle.com/datasets/beschue/industrial-classification-data-set)

**Actividades**

• **1.** **Transfer** **learning** **con** **VGG16:** congelar la base
convolucional, añadir una cabeza clasificadora propia y entrenarla.
Opcionalmente, aplicar fine-tuning de las últimas capas.

• **2.** **Transfer** **learning** **con** **otra** **arquitectura:**
elegir cualquier otra arquitectura preentrenada, no necesariamente con
pesos de ImageNet (por ejemplo ResNet, MobileNet, EfficientNet o

> ViT), y adaptarla al problema.

• **3.** **Red** **desde** **cero:** diseñar y entrenar una CNN propia
sin pesos preentrenados.

**Requisitos**

• Aplicar lo explicado en el libro de Andrew Ng: partición
train/dev/test, análisis de sesgo y varianza, regularización, data
augmentation, ajuste de hiperparámetros y análisis de errores.

• Comparar los tres modelos con métricas apropiadas (accuracy,
precisión, recall, F1, matriz de confusión) y curvas de entrenamiento.

• Analizar los errores de cada enfoque y discutir cuál es más adecuado y
por qué.

**Entregables**

Notebook documentado y breve informe (máximo 2 páginas) con resultados,
comparación y conclusiones.

**Criterios** **de** **evaluación**

• Correcta implementación: 40 %

• Rigor experimental y aplicación de las prácticas de Ng: 30 % •
Análisis y conclusiones: 30 %
