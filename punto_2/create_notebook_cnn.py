"""Genera punto_2_cnn.ipynb — versión CONVOLUCIONAL del punto 2 (contraste vs MLP denso)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nb.metadata["language_info"] = {"name": "python"}

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

# 1 — título
c1 = """# Parcial DL — Punto 2 (versión CONVOLUCIONAL)
**Lengua de señas 27 clases — CNN vs MLP**

Este cuaderno **varía** el `punto_2.ipynb` original (MLP denso con `Flatten`):
- Mantiene mismo dataset (`ardamavi/27-class-sign-language-dataset` → `X.npy`, `Y.npy`), mismo split 70/15/15 y misma resolución 64×64 gris para comparación justa.
- Cambia el modelo: **CNN con 3 bloques Conv2D+BN+MaxPool** en vez de solo `Dense`. Así se conserva la estructura espacial (vecinos, bordes de dedos) y se usan menos parámetros.
- Añade **Data Augmentation** (rotación/shift/zoom) y visualización de filtros + errores típicos.

> Colab: usa GPU (Entorno → Cambiar tipo de entorno → T4 GPU). En CPU tarda mucho.
"""

# 2 — setup
c2 = """# 0. Setup — backend Keras + GPU check
import os
# En Colab con TF no hace falta KERAS_BACKEND=torch. Lo dejamos en TF (más rápido para CNN).
# Si tu repo exige torch, descomenta la línea siguiente ANTES de importar keras:
# os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import cv2
import kagglehub

import keras
from keras.models import Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense,
                          Dropout, BatchNormalization, RandomFlip,
                          RandomRotation, RandomZoom, RandomTranslation,
                          Rescaling)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("Keras:", keras.__version__)
import tensorflow as tf
print("TF:", tf.__version__, "| GPU:", tf.config.list_physical_devices('GPU') or "CPU (en Colab activa GPU!)")
"""

c3 = """## 1. Descarga del dataset
`X.npy` son fotos RGB 128×128, `Y.npy` etiquetas. No necesitan login si es público.
"""
c4 = """path = kagglehub.dataset_download("ardamavi/27-class-sign-language-dataset")
print("Ruta:", path)
X_raw = np.load(os.path.join(path, "X.npy"))
y_raw = np.load(os.path.join(path, "Y.npy"))
print("X_raw:", X_raw.shape, X_raw.dtype, "| ej. min/max:", X_raw[0].min(), X_raw[0].max())
print("y_raw:", y_raw.shape, "| clases únicas:", np.unique(y_raw.flatten())[:10], "... total:", len(np.unique(y_raw.flatten())))
# Muestra 1 imagen cruda
plt.figure(figsize=(2.5,2.5)); plt.imshow(X_raw[0]); plt.title(f"Cruda {X_raw[0].shape} label={y_raw.flatten()[0]}"); plt.axis("off"); plt.show()
"""

c5 = """## 2. Preprocesamiento PARA CNN (¡aquí está la diferencia!)
**MLP hacía:** gris 64×64 → `flatten()` → vector 4096, pierde vecinos.
**CNN hace:** gris 64×64 → tensor `(64,64,1)` normalizado [0,1], conserva filas/columnas para que el kernel 3×3 vea bordes de dedos.
Split estratificado 70/15/15 igual que el denso para comparar.
"""
c6 = """IMG_SIZE = 64  # igual que el MLP para comparación justa

def preprocesar_para_cnn(X_raw):
    # RGB 128x128 -> gris 64x64x1 float32 [0,1]. Conserva forma espacial.
    # NOTA: el .npy ya viene float32 0-1, asi que solo dividimos si viene 0-255.
    N = len(X_raw)
    X_out = np.empty((N, IMG_SIZE, IMG_SIZE, 1), dtype='float32')
    # Detecta rango una vez (evita doble /255 que dejaría todo negro)
    rmax = float(np.max(X_raw[:50]))  # mira 50 para no cargar todo
    div = 255.0 if rmax > 1.5 else 1.0
    print(f"Rango detectado max~{rmax:.3f} → div={div}")
    for i in range(N):
        # cv2 espera uint8 o float32; si es float 0-1 funciona directo
        g = cv2.cvtColor(X_raw[i], cv2.COLOR_RGB2GRAY)  # (128,128)
        r = cv2.resize(g, (IMG_SIZE, IMG_SIZE))          # (64,64)
        X_out[i, :, :, 0] = r.astype('float32') / div
    return X_out

X = preprocesar_para_cnn(X_raw)
print("X CNN:", X.shape)  # (N,64,64,1)

le = LabelEncoder()
y = le.fit_transform(y_raw.flatten())
clases = list(le.classes_)
print("Clases:", clases, "| num:", len(clases))

# Split 70/15/15 estratificado (misma semilla que MLP)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)
print(f"Train {X_train.shape[0]} | Val {X_val.shape[0]} | Test {X_test.shape[0]}")

# Mosaico 3×5 de ejemplos train
fig, ax = plt.subplots(3,5, figsize=(8,5))
for i, a in enumerate(ax.flat):
    a.imshow(X_train[i,:,:,0], cmap="gray"); a.set_title(str(clases[y_train[i]]), fontsize=9); a.axis("off")
plt.suptitle("Ejemplos train 64×64 gris (entrada CNN)"); plt.tight_layout(); plt.show()
"""

c7 = """## 3. Modelo CNN (3 bloques)
Cada bloque = `Conv3×3 same + ReLU + BN + MaxPool2×2 + Dropout`.
- Bloque1 32 filtros: bordes de dedos.
- Bloque2 64: texturas/falanges.
- Bloque3 128: partes de mano/seña.
- Head: `Flatten 8×8×128=8192 → Dense256 → Dropout0.5 → Softmax27`.
MaxPool tiene **0 params** (solo max 2×2), por eso la CNN usa menos pesos que la densa de 1024.
"""
c8 = """num_classes = len(clases)

# Aumento de datos INSIDE-model (solo en train, en inferencia se desactiva solo)
augment = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.08),       # ±~29°
    RandomTranslation(0.1, 0.1),
    RandomZoom(0.1),
], name="augment")

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Rescaling(1.0),  # ya está [0,1], se deja explícito
    augment,

    Conv2D(32, 3, padding="same", activation="relu", name="conv1"),
    BatchNormalization(),
    MaxPooling2D(2, name="pool1"),   # 64→32
    Dropout(0.25),

    Conv2D(64, 3, padding="same", activation="relu", name="conv2"),
    BatchNormalization(),
    MaxPooling2D(2, name="pool2"),   # 32→16
    Dropout(0.25),

    Conv2D(128, 3, padding="same", activation="relu", name="conv3"),
    BatchNormalization(),
    MaxPooling2D(2, name="pool3"),   # 16→8
    Dropout(0.3),

    Flatten(name="flatten"),         # 8*8*128=8192
    Dense(256, activation="relu", name="fc1"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation="softmax", name="pred")
])

model.summary()
print("\\nParams totales:", f"{model.count_params():,}")
# Compila con sparse (y enteros, no one-hot)
model.compile(optimizer=Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
"""

c9 = """## 4. Entrenamiento
`EarlyStopping(patience=12)` + `ReduceLROnPlateau`. En Colab-GPU ~30-50 épocas. En CPU baja `epochs` o usa `SMOKE_TEST`.
"""
c10 = """SMOKE_TEST = False  # pon True para probar en 2 min (subset 2000 + 3 épocas)
if SMOKE_TEST:
    idx = np.random.RandomState(0).choice(len(X_train), 2000, replace=False)
    Xtr, ytr = X_train[idx], y_train[idx]
    idxv = np.random.RandomState(1).choice(len(X_val), 400, replace=False)
    Xv, yv = X_val[idxv], y_val[idxv]
    EPOCHS, BS = 3, 64
else:
    Xtr, ytr, Xv, yv = X_train, y_train, X_val, y_val
    EPOCHS, BS = 50, 128

cbs = [EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
       ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)]

history = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=EPOCHS, batch_size=BS, callbacks=cbs)
"""

c11 = """## 5. Curvas + Evaluación en test
"""
c12 = """fig, (a1,a2) = plt.subplots(1,2, figsize=(12,4))
a1.plot(history.history["loss"], label="train"); a1.plot(history.history["val_loss"], label="val")
a1.set_title("Loss"); a1.set_xlabel("época"); a1.legend(); a1.grid(alpha=0.3)
a2.plot(history.history["accuracy"], label="train"); a2.plot(history.history["val_accuracy"], label="val")
a2.set_title("Accuracy"); a2.set_xlabel("época"); a2.legend(); a2.grid(alpha=0.3)
plt.show()

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss {loss:.4f} | Test acc {acc*100:.2f}%")
yp = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, yp, target_names=[str(c) for c in clases]))

cm = confusion_matrix(y_test, yp)
plt.figure(figsize=(9,7)); sns.heatmap(cm, annot=False, cmap="Blues")
plt.xlabel("pred"); plt.ylabel("true"); plt.title("Matriz confusión — CNN"); plt.show()
"""

c13 = """## 6. Qué aprendió la CNN (filtros + errores)
Visualizamos 8 filtros del `conv1` y 8 feature maps de una muestra, + 5 errores típicos. Esto NO podía hacerlo el MLP porque aplanaba.
"""
c14 = """# 6a. Filtros conv1 (32 kernels 3×3×1)
W = model.get_layer("conv1").get_weights()[0]  # (3,3,1,32)
fig, ax = plt.subplots(2,4, figsize=(8,4))
for i,a in enumerate(ax.flat):
    a.imshow(W[:,:,0,i], cmap="seismic", vmin=-W.max(), vmax=W.max()); a.set_title(f"f{i}", fontsize=9); a.axis("off")
plt.suptitle("Filtros conv1 3×3 (bordes)"); plt.tight_layout(); plt.show()

# 6b. Feature maps de una muestra test
from keras.models import Model as KModel
fmap_model = KModel(inputs=model.input, outputs=[model.get_layer(n).output for n in ["conv1","conv2","conv3"]])
# OJO: augment se desactiva en inferencia (training=False por defecto en predict)
sample = X_test[:1]
f1,f2,f3 = fmap_model.predict(sample, verbose=0)
for name, f in zip(["conv1 32×32","conv2 16×16","conv3 8×8"], [f1,f2,f3]):
    fig, ax = plt.subplots(2,4, figsize=(8,4))
    for i,a in enumerate(ax.flat):
        ch = f[0,:,:,i]; v1,v99 = np.percentile(ch,(1,99))
        a.imshow(ch, cmap="viridis", vmin=v1, vmax=v99 if v99>v1 else None); a.set_title(f"c{i}", fontsize=8); a.axis("off")
    plt.suptitle(f"Feature maps {name}"); plt.tight_layout(); plt.show()

# 6c. 5 errores (pred != true)
err = np.where(yp != y_test)[0][:5]
fig, ax = plt.subplots(1,5, figsize=(10,2.5))
for a, idx in zip(ax, err):
    a.imshow(X_test[idx,:,:,0], cmap="gray"); a.set_title(f"T:{clases[y_test[idx]]}\\nP:{clases[yp[idx]]}", fontsize=8); a.axis("off")
plt.suptitle("Errores típicos (señas parecidas)"); plt.tight_layout(); plt.show()

model.save("modelo_punto2_cnn.keras")
print("Guardado modelo_punto2_cnn.keras")
"""

c15 = """## 7. Conclusiones CNN vs MLP
1. **Espacialidad:** CNN conserva vecinos con kernels 3×3; MLP los rompe con flatten → CNN distingue dedos/falanges mejor.
2. **Parámetros:** primera densa del MLP `4096×1024≈4.2M` solo ahí; CNN `conv1 (3×3×1+1)×32=320`. Menos overfit, menos memoria.
3. **Augment + Pool:** Translación/rotación + MaxPool(0 params) dan invariancia a posición/tamaño de mano, clave en señas.
4. **Esperado:** CNN supera al MLP en test-acc con menos épocas (típico +8-15 pts en este dataset 27 clases). Si no, revisar lr/augment.
"""

nb.cells = [md(c1), code(c2), md(c3), code(c4), md(c5), code(c6), md(c7), code(c8),
            md(c9), code(c10), md(c11), code(c12), md(c13), code(c14), md(c15)]
with open("/tmp/opencode/punto_2_cnn.ipynb","w",encoding="utf-8") as f:
    nbf.write(nb,f)
print("OK punto_2_cnn.ipynb")
