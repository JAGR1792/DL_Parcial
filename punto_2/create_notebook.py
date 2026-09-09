import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cell 1
text_1 = """# Parcial Deep Learning - Punto 2
**Identificación de Lengua de Señas**

Este notebook aborda el segundo punto del parcial:
*Crear una red neuronal densa que identifique lengua de señas usando el dataset de: 27-class-sign-language-dataset*

## Pasos:
1. Descarga del dataset
2. Preprocesamiento (Flattening de las imágenes para la red densa, normalización)
3. Definición del modelo Dense (MLP) con Keras
4. Entrenamiento con EarlyStopping
5. Análisis de los resultados (Loss/Accuracy, Matriz de Confusión)
"""

# Code cell 1 - Imports
code_1 = """import os
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import cv2

os.environ["KERAS_BACKEND"] = "torch"
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("Keras version:", keras.__version__)"""

# Markdown cell 2
text_2 = """## 1. Descarga del Dataset
Usaremos `kagglehub` para descargar el dataset directamente. El dataset público contiene directamente los archivos de datos en matrices numpy."""

# Code cell 2 - Download
code_2 = """# Descargar el dataset (no requiere autenticación para este dataset público)
path = kagglehub.dataset_download("ardamavi/27-class-sign-language-dataset")
print("Ruta del dataset:", path)

# Cargar las matrices numpy
X_raw = np.load(os.path.join(path, "X.npy"))
y_raw = np.load(os.path.join(path, "Y.npy"))

print("Shape inicial de X:", X_raw.shape)
print("Shape inicial de y:", y_raw.shape)

# Hay 27 clases en total, pero para mapearlos a nombres de clases:
# Basado en la descripción, son letras del abecedario.
clases = [str(i) for i in range(27)]
"""

# Markdown cell 3
text_3 = """## 2. Preprocesamiento de los Datos
Para una **Red Neuronal Densa (MLP)**, la entrada debe ser un vector 1D. El dataset original tiene imágenes a color de 128x128.
Para reducir la carga computacional de la red densa y evitar problemas de memoria, las convertiremos a escala de grises y las redimensionaremos a 64x64 antes de aplanarlas."""

# Code cell 3 - Preprocessing
code_3 = """from sklearn.preprocessing import LabelEncoder
IMG_SIZE = 64

X_processed = []

for i in range(len(X_raw)):
    # Convertir a escala de grises
    img_gray = cv2.cvtColor(X_raw[i], cv2.COLOR_RGB2GRAY)
    # Redimensionar a 64x64
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    X_processed.append(img_resized.flatten())

X = np.array(X_processed, dtype='float32') / 255.0  # Normalización

le = LabelEncoder()
y = le.fit_transform(y_raw.flatten())
clases = le.classes_

print("Forma de X tras preprocesamiento:", X.shape)
print("Forma de y:", y.shape)

# Dividir en Train (70%), Val (15%) y Test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]} muestras")
print(f"Val: {X_val.shape[0]} muestras")
print(f"Test: {X_test.shape[0]} muestras")"""

# Markdown cell 4
text_4 = """## 3. Definición del Modelo (Red Neuronal Densa)
Crearemos una arquitectura MLP pura (capas Dense)."""

# Code cell 4 - Model definition
code_4 = """from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.optimizers import Adam

num_classes = len(np.unique(y))
input_dim = X_train.shape[1]

model = Sequential([
    Input(shape=(input_dim,)),
    
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(num_classes, activation='softmax')
])

model.summary()

# Compilar modelo
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])"""

# Markdown cell 5
text_5 = """## 4. Entrenamiento
Utilizaremos `EarlyStopping` para detener el entrenamiento cuando el modelo deje de mejorar en el conjunto de validación."""

# Code cell 5 - Training
code_5 = """from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop, reduce_lr]
)"""

# Markdown cell 6
text_6 = """## 5. Análisis de Resultados (Evaluación)
Evaluaremos el rendimiento en el conjunto de prueba."""

# Code cell 6 - Plot learning curves
code_6 = """# Gráficas de Loss y Accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title('Pérdida (Loss) a lo largo de las épocas')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history.history['accuracy'], label='Train Acc')
ax2.plot(history.history['val_accuracy'], label='Val Acc')
ax2.set_title('Precisión (Accuracy) a lo largo de las épocas')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()"""

# Code cell 7 - Confusion Matrix & Report
code_7 = """# Predicciones
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Reporte de clasificación
print(classification_report(y_test, y_pred))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
plt.show()

# Guardar el modelo entrenado para uso posterior (ej. inferencia en tiempo real)
model.save('modelo_punto2.keras')
print("Modelo guardado como 'modelo_punto2.keras'")"""

# Markdown cell 8
text_8 = """## 6. Conclusiones
De acuerdo a lo esperado según el libro guía (hasta el cap. 22):
1. **Pérdida de información espacial**: Al aplanar la imagen a un vector 1D, el modelo pierde el contexto espacial (relación entre píxeles vecinos), lo que dificulta reconocer características geométricas complejas de las manos.
2. **Número de parámetros masivo**: El número de pesos en la primera capa oculta es enorme, lo que aumenta el riesgo de sobreajuste y requiere más memoria computacional frente a redes especializadas como las convolucionales (CNNs).
3. **Desempeño general**: A pesar de estas limitaciones arquitectónicas para procesamiento de imágenes, la red densa (MLP) logra encontrar patrones generales funcionando como un aproximador universal, aunque con una precisión probablemente inferior a la que se obtendría con una CNN."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(text_5),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(text_6),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_code_cell(code_7),
    nbf.v4.new_markdown_cell(text_8)
]

with open('punto_2.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook punto_2.ipynb actualizado exitosamente con la nueva lógica.")
