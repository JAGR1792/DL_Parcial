# -*- coding: utf-8 -*-
"""
Genera el notebook del Parcial 2 de Deep Learning.

Tema: clasificación de imágenes industriales (Industrial Dataset, Kaggle)
       comparando tres estrategias:
         A. Transfer learning con VGG16 (base congelada + cabeza propia + fine-tuning)
         B. Transfer learning con ResNet50V2
         C. CNN entrenada desde cero

Prácticas del libro de Andrew Ng aplicadas:
  - partición train / dev / test
  - análisis de sesgo y varianza (curvas de aprendizaje)
  - regularización (dropout, L2, label smoothing, early stopping)
  - data augmentation
  - ajuste de hiperparámetros sobre dev
  - análisis de errores sobre test

Dataset: estructura PLANA (sin subcarpetas por clase).
  Train_Dataset/images/adapter_plate_triangular000000.jpg
  La clase se extrae del nombre del archivo (quitar dígitos finales).

Uso:
    python build_notebook.py
Genera: Parcial2_Deep_Learning.ipynb
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "Parcial2_Deep_Learning.ipynb"

CELLS = []


def md(text: str) -> None:
    CELLS.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


def code(text: str) -> None:
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


# ---------------------------------------------------------------------------
# Portada
# ---------------------------------------------------------------------------
md(
    r"""
# Parcial 2 — Deep Learning
## Clasificación de imágenes industriales: transfer learning vs. red desde cero

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**Instrucciones para correr en la web de Colab o Kaggle:**

1. `Runtime > Change runtime type > T4 GPU` (o L4 si tienes Colab Pro).
2. En Colab: panel izquierdo → ícono de la **llave (Secrets)** → agregar `KAGGLE_API_TOKEN` con tu token `KGAT_...` y activar "Notebook access".
   En Kaggle: el dataset se monta automáticamente en `/kaggle/input/`.
3. `Runtime > Run all` (o ejecutar celda por celda).
4. Los checkpoints y resultados se guardan en `/kaggle/working/out` (Kaggle) o Google Drive (Colab).

**Integrantes:** J. A. García Romero · S. J. Rodríguez Gutiérrez · C. A. Díaz García

**Dataset:** [Industrial Dataset](https://www.kaggle.com/datasets/beschue/industrial-classification-data-set)
(MVTec ITODD renderizado con SynthiCAD). 10 clases de piezas industriales,
100 000 imágenes RGB de 224×224, repartidas en 70 000 train / 10 000 valid / 20 000 test.

> **Nota sobre la estructura:** el dataset tiene las imágenes en carpetas planas
> (`Train_Dataset/images/adapter_plate_triangular000000.jpg`). La clase se extrae
> del nombre del archivo (se quitan los dígitos finales).

### Objetivo

Comparar tres estrategias de modelado para clasificar piezas industriales:

| # | Estrategia | Idea |
|---|---|---|
| **A** | Transfer learning con **VGG16** | congelar la base convolucional, entrenar una cabeza propia y, opcionalmente, hacer fine-tuning de las últimas capas |
| **B** | Transfer learning con **ResNet50V2** | otra arquitectura preentrenada en ImageNet, adaptada al problema |
| **C** | **CNN desde cero** | red propia sin pesos preentrenados |

### Prácticas de Andrew Ng que se aplican

1. **Partición train / dev / test** y uso disciplinado: dev para decidir, test solo al final.
2. **Análisis de sesgo y varianza** a partir de los errores de train y dev, más curvas de aprendizaje.
3. **Regularización**: dropout, L2, label smoothing, early stopping y reducción de learning rate.
4. **Data augmentation**: volteo, recorte aleatorio y jitter fotométrico.
5. **Ajuste de hiperparámetros** sobre dev (no sobre test).
6. **Análisis de errores**: matriz de confusión, pares confundidos y galería de fallos.

> **Regla de oro:** el conjunto de test no se toca hasta la sección final de evaluación.
"""
)

# ---------------------------------------------------------------------------
# 0. Configuración
# ---------------------------------------------------------------------------
md(
    r"""
## 0. Configuración, reproducibilidad y entorno

Fijamos semillas, activamos precisión mixta (la T4 de Colab lo agradece) y definimos
las rutas de trabajo. Los artefactos se guardan en la carpeta de salida para sobrevivir
a una desconexión.
"""
)

code(
    r"""
# ============================================================
# 0.0 Verificación de entorno (Colab / Kaggle web)
# ============================================================
import importlib, subprocess, sys

for pkg in ["kagglehub", "tensorflow", "keras"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs disponibles:", gpus)
if not gpus:
    print("ADVERTENCIA: no hay GPU. En Colab: Runtime > Change runtime type > T4 GPU.")
else:
    print("GPU OK:", tf.config.experimental.get_device_details(gpus[0]).get("device_name"))

# Detectar entorno
from pathlib import Path
if Path("/kaggle/input").exists():
    print("Entorno: Kaggle")
else:
    print("Entorno: Colab / local")
"""
)

code(
    r"""
# ============================================================
# 0. Configuración, semillas y entorno
# ============================================================
import os, sys, json, time, random, shutil, zipfile, subprocess
from pathlib import Path
from collections import Counter, defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras
from keras import layers, regularizers, callbacks

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

keras.mixed_precision.set_global_policy("mixed_float16")

print("Python     :", sys.version.split()[0])
print("TensorFlow :", tf.__version__)
print("Keras      :", keras.__version__)
print("Política   :", keras.mixed_precision.global_policy().name)
"""
)

code(
    r"""
# ============================================================
# Parámetros globales del experimento
# ============================================================
IMG_SIZE   = 224
BATCH_SIZE = 64
NUM_CLASSES = 10

EPOCHS_TUNING = 3
EPOCHS_VGG    = 10
EPOCHS_FT     = 3
EPOCHS_RESNET = 10
EPOCHS_CNN    = 15

TUNING_FRAC = 0.20
RUN_TUNING  = True
RUN_LEARNING_CURVE = True

FAST_DEV_RUN = False
FAST_FRAC    = 0.05

# Error óptimo asumido (human-level) para estas piezas: ~0 %
OPTIMAL_ERROR_RATE = 0.0

AUTOTUNE = tf.data.AUTOTUNE

print(f"IMG_SIZE={IMG_SIZE} | BATCH_SIZE={BATCH_SIZE} | mixed_float16 | seed={SEED}")
print("FAST_DEV_RUN =", FAST_DEV_RUN)
"""
)

code(
    r"""
# ============================================================
# Directorio de salida
# ============================================================
from pathlib import Path

if Path("/kaggle/input").exists():
    OUT_DIR = Path("/kaggle/working/out")
    print("Entorno Kaggle: salida en", OUT_DIR)
elif Path("data/DataSet").exists():
    OUT_DIR = Path("out")
    print("Entorno local: salida en", OUT_DIR.resolve())
else:
    OUT_DIR = Path("/content/out")
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT_DIR = Path("/content/drive/MyDrive/Parcial2_DL")
        print("Drive montado:", OUT_DIR)
    except Exception as exc:
        print("No se pudo montar Drive:", exc)
        OUT_DIR = Path("/content/out")

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Artefactos ->", OUT_DIR)
"""
)

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
md(
    r"""
## 1. Obtención del dataset

El dataset tiene una estructura **plana**: las imágenes están en `Train_Dataset/images/`
sin subcarpetas por clase. La clase se extrae del nombre del archivo
(`adapter_plate_triangular000000.jpg` → "adapter plate triangular").

En Kaggle el dataset se monta automáticamente en `/kaggle/input/`.
En Colab se descarga con `kagglehub` usando el secreto `KAGGLE_API_TOKEN`.
"""
)

code(
    r"""
# ============================================================
# 1. Rutas y carga del dataset (estructura plana)
# ============================================================
import os, re
from pathlib import Path

if Path("/kaggle/input").exists():
    DATA_ROOT = Path("/kaggle/input/industrial-classification-data-set/DataSet")
    print("Dataset en Kaggle:", DATA_ROOT)
elif Path("data/DataSet").exists():
    DATA_ROOT = Path("data/DataSet")
    print("Dataset local encontrado:", DATA_ROOT.resolve())
else:
    import kagglehub
    src = Path(kagglehub.dataset_download("beschue/industrial-classification-data-set"))
    DATA_ROOT = src / "DataSet"
    print("Dataset descargado en:", DATA_ROOT)

SPLITS = {"train": "Train_Dataset", "valid": "Valid_Dataset", "test": "Test_Dataset"}

# Clases del dataset (orden alfabético)
CLASS_NAMES = [
    "adapter plate triangular",
    "bracket big",
    "clamp small",
    "engine part cooler round",
    "engine part cooler square",
    "injection pump",
    "screw",
    "star",
    "tee connector",
    "thread",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_MAP = {cn.replace(" ", "_"): i for i, cn in enumerate(CLASS_NAMES)}


def filename_to_class(filename):
    name = os.path.splitext(filename)[0]
    prefix = re.sub(r"\d+$", "", name)
    assert prefix in CLASS_MAP, f"Clase desconocida: {filename} (prefijo: {prefix})"
    return CLASS_MAP[prefix]


def list_images_and_labels(split_name):
    folder = DATA_ROOT / SPLITS[split_name] / "images"
    paths, labels = [], []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(p))
            labels.append(filename_to_class(p.name))
    return paths, labels


# Verificar conteos
for split in ["train", "valid", "test"]:
    paths, labels = list_images_and_labels(split)
    from collections import Counter
    c = Counter(labels)
    print(f"{split:5s}: {len(paths)} imágenes, {len(c)} clases")
    assert len(paths) > 0

print("\nClases:", CLASS_NAMES)
"""
)

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
md(
    r"""
## 2. Análisis exploratorio

Distribución de clases (balanceada: 7 000 por clase en train) y ejemplos visuales.
"""
)

code(
    r"""
# ============================================================
# 2. Distribución de clases
# ============================================================
from collections import Counter
import pandas as pd

COUNTS = {}
for split in ["train", "valid", "test"]:
    paths, labels = list_images_and_labels(split)
    COUNTS[split] = {CLASS_NAMES[i]: sum(1 for l in labels if l == i) for i in range(NUM_CLASSES)}

balance = pd.DataFrame(COUNTS).rename_axis("clase").reset_index()
balance["total"] = balance[["train", "valid", "test"]].sum(axis=1)
display(balance)

fig, ax = plt.subplots(figsize=(11, 4))
balance.set_index("clase")[["train", "valid", "test"]].plot(
    kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52"], width=0.8
)
ax.set_title("Distribución de imágenes por clase y partición")
ax.set_ylabel("N.º de imágenes")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="Partición")
plt.tight_layout()
plt.show()
"""
)

code(
    r"""
# ============================================================
# 2.1 Galería de ejemplos (2 por clase)
# ============================================================
from PIL import Image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(NUM_CLASSES, 2, figsize=(5, 2.2 * NUM_CLASSES))
for row, class_name in enumerate(CLASS_NAMES):
    prefix = class_name.replace(" ", "_")
    class_imgs = [p for p in (DATA_ROOT / "Train_Dataset" / "images").iterdir()
                  if p.name.startswith(prefix)][:2]
    for col in range(2):
        ax = axes[row, col]
        if col < len(class_imgs):
            ax.imshow(Image.open(class_imgs[col]).convert("RGB"))
        ax.axis("off")
        if col == 0:
            ax.set_title(class_name, loc="left", fontsize=9)
plt.suptitle("Ejemplos del conjunto de entrenamiento", y=1.005)
plt.tight_layout()
plt.show()
"""
)

# ---------------------------------------------------------------------------
# 3. Pipeline
# ---------------------------------------------------------------------------
md(
    r"""
## 3. Partición train / dev / test y pipeline `tf.data`

El dataset ya trae tres particiones. Las mapeamos a la nomenclatura de Ng:

| Partición del dataset | Rol | Tamaño |
|---|---|---|
| `Train_Dataset` | entrenamiento | 70 000 |
| `Valid_Dataset` | **dev** (decisiones, hiperparámetros) | 10 000 |
| `Test_Dataset` | **test** (evaluación final, una sola vez) | 20 000 |

> **Principios de Ng aplicados (caps. 5-12):** dev y test provienen de la
> misma distribución (cap. 6); dev de 10 000 ejemplos permite detectar mejoras
> del 0.1 % (cap. 7); se usa una métrica única para decidir (caps. 8-9); el test
> no se toca hasta el final (cap. 11).

**Decisiones del pipeline**

* Las imágenes están en una carpeta plana; la etiqueta se extrae del nombre del archivo.
* **Data augmentation** solo en train: volteo horizontal, recorte aleatorio (85–100 % del área)
  con reescalado, y jitter de brillo, contraste y saturación.
* Cada arquitectura usa su propio `preprocess_input`.
* `prefetch` + `num_parallel_calls` para que la GPU no espere a la CPU.
"""
)

code(
    r"""
# ============================================================
# 3. Pipeline de datos (estructura plana)
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE

def augment_single(image, label):
    image = tf.image.random_flip_left_right(image)
    scale = tf.random.uniform([], 0.85, 1.0)
    new_size = tf.cast(tf.cast(IMG_SIZE, tf.float32) * scale, tf.int32)
    image = tf.image.random_crop(image, [new_size, new_size, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.image.random_brightness(image, max_delta=25.0)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.90, upper=1.10)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def preprocess_vgg(image, label):
    return keras.applications.vgg16.preprocess_input(image), label


def preprocess_resnet(image, label):
    return keras.applications.resnet_v2.preprocess_input(image), label


def preprocess_scratch(image, label):
    return image / 255.0, label


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, tf.one_hot(label, NUM_CLASSES)


def build_flat_pipeline(split_name, preprocess_fn, training=False, fast=False):
    paths, labels = list_images_and_labels(split_name)
    if fast and training:
        n = int(len(paths) * FAST_FRAC)
        paths = paths[:n]
        labels = labels[:n]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)

    def _map(path, label):
        img, onehot = load_image(path, label)
        if training:
            img, onehot = augment_single(img, onehot)
        img, onehot = preprocess_fn(img, onehot)
        return img, onehot

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# Construir los 3 pipelines
train_vgg = build_flat_pipeline("train", preprocess_vgg, training=True, fast=FAST_DEV_RUN)
dev_vgg   = build_flat_pipeline("valid", preprocess_vgg)
test_vgg  = build_flat_pipeline("test", preprocess_vgg)

train_res = build_flat_pipeline("train", preprocess_resnet, training=True, fast=FAST_DEV_RUN)
dev_res   = build_flat_pipeline("valid", preprocess_resnet)
test_res  = build_flat_pipeline("test", preprocess_resnet)

train_scr = build_flat_pipeline("train", preprocess_scratch, training=True, fast=FAST_DEV_RUN)
dev_scr   = build_flat_pipeline("valid", preprocess_scratch)
test_scr  = build_flat_pipeline("test", preprocess_scratch)

# Pipelines sin augmentation (para error de train, estilo Ng)
train_eval_vgg = build_flat_pipeline("train", preprocess_vgg, training=False, fast=FAST_DEV_RUN)
train_eval_res = build_flat_pipeline("train", preprocess_resnet, training=False, fast=FAST_DEV_RUN)
train_eval_scr = build_flat_pipeline("train", preprocess_scratch, training=False, fast=FAST_DEV_RUN)

# TEST_PATHS para galería de errores
test_paths, _ = list_images_and_labels("test")
TEST_PATHS = test_paths

print("Lotes por época:")
print("  train VGG   :", len(train_vgg))
print("  train ResNet:", len(train_res))
print("  train CNN   :", len(train_scr))
print("  dev         :", len(dev_vgg))
print("  test        :", len(test_vgg))
"""
)

code(
    r"""
# ============================================================
# 3.1 Verificación visual del augmentation
# ============================================================
def show_augmented_examples(n=6):
    paths, labels = list_images_and_labels("train")
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8))
    for i in range(n):
        img = tf.io.read_file(paths[i])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        axes[0, i].imshow(tf.cast(img, tf.uint8).numpy())
        axes[0, i].set_title(CLASS_NAMES[labels[i]], fontsize=8)
        axes[0, i].axis("off")

        aug, _ = augment_single(img, tf.one_hot(labels[i], NUM_CLASSES))
        axes[1, i].imshow(tf.cast(tf.clip_by_value(aug, 0, 255), tf.uint8).numpy())
        axes[1, i].set_title("aumentada", fontsize=8)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("augmented")
    plt.tight_layout()
    plt.show()

show_augmented_examples()
"""
)

# ---------------------------------------------------------------------------
# 4. Utilidades
# ---------------------------------------------------------------------------
md(
    r"""
## 4. Utilidades de entrenamiento y evaluación

Centralizamos aquí lo que se repite en los tres modelos: callbacks, curvas de entrenamiento,
métricas (accuracy, precisión, recall, F1 macro) y el registro de resultados.
"""
)

code(
    r"""
# ============================================================
# 4. Utilidades
# ============================================================
RESULTS = {}


def make_callbacks(name, patience=2):
    ckpt = OUT_DIR / f"{name}.keras"
    return [
        callbacks.ModelCheckpoint(str(ckpt), monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1,
                                    min_lr=1e-6, verbose=1),
    ], ckpt


def plot_history(history, title):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist["loss"], "o-", label="train")
    axes[0].plot(epochs, hist["val_loss"], "s-", label="dev")
    axes[0].set_title(f"{title} — pérdida")
    axes[0].set_xlabel("época"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, hist["accuracy"], "o-", label="train")
    axes[1].plot(epochs, hist["val_accuracy"], "s-", label="dev")
    axes[1].set_title(f"{title} — accuracy")
    axes[1].set_xlabel("época"); axes[1].set_ylabel("accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_model(model, dataset, name, store=True):
    t0 = time.time()
    y_prob = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in dataset])
    y_true = y_true[: len(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    try:
        top3 = top_k_accuracy_score(y_true, y_prob, k=3, labels=list(range(NUM_CLASSES)))
    except Exception:
        top3 = float("nan")

    result = {
        "modelo": name,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_weighted": f1_w,
        "top3_accuracy": top3,
        "tiempo_eval_s": round(time.time() - t0, 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"accuracy   : {acc:.4f}")
    print(f"F1 macro   : {f1:.4f}")
    print(f"F1 weighted: {f1_w:.4f}")
    print(f"top-3      : {top3:.4f}")
    print(f"tiempo     : {result['tiempo_eval_s']} s")

    if store:
        RESULTS[name] = result
    return result


def bias_variance_report(model, name, train_ds, dev_ds, sample_train=10000,
                          optimal_error=OPTIMAL_ERROR_RATE):
    train_sample = train_ds.unbatch().take(sample_train).batch(BATCH_SIZE)
    tr_loss, tr_acc = model.evaluate(train_sample, verbose=0)
    dv_loss, dv_acc = model.evaluate(dev_ds, verbose=0)

    tr_err = 1 - tr_acc
    dv_err = 1 - dv_acc
    avoidable_bias = max(0.0, tr_err - optimal_error)
    variance = max(0.0, dv_err - tr_err)
    total_err = dv_err

    if avoidable_bias > 0.05 and avoidable_bias >= variance:
        verdict = "SESGO ALTO (avoidable bias): no ajusta bien ni el train."
        action = ("Ng cap.25: aumentar capacidad (mas capas/neuronas), reducir regularizacion, "
                  "mejorar features segun error analysis, o cambiar arquitectura.")
    elif variance > 0.05 and variance > avoidable_bias:
        verdict = "VARIANZA ALTA (overfitting): gran brecha train-dev."
        action = ("Ng cap.27: mas datos, regularizacion (L2/dropout), early stopping, "
                  "o data augmentation mas agresiva.")
    elif total_err <= optimal_error + 0.02:
        verdict = "BUEN AJUSTE: cerca del error optimo."
        action = "El cuello de botella esta en los datos o en el techo del modelo."
    else:
        verdict = "AJUSTE RAZONABLE: bias y varianza moderados."
        action = "Mejorar features o revisar la distribucion de los datos."

    print(f"{name}:")
    print(f"   error optimo (human-level) : {optimal_error:.4f}")
    print(f"   train_acc / train_err      : {tr_acc:.4f} / {tr_err:.4f}")
    print(f"   dev_acc   / dev_err       : {dv_acc:.4f} / {dv_err:.4f}")
    print(f"   avoidable bias            : {avoidable_bias:.4f}")
    print(f"   variance                  : {variance:.4f}")
    print(f"   -> {verdict}")
    print(f"   -> {action}")

    return {"modelo": name, "optimal_error": optimal_error,
            "train_acc": tr_acc, "dev_acc": dv_acc,
            "train_err": tr_err, "dev_err": dv_err,
            "avoidable_bias": avoidable_bias, "variance": variance,
            "diagnostico": verdict, "accion": action}
"""
)

# ---------------------------------------------------------------------------
# 5. Tuning
# ---------------------------------------------------------------------------
md(
    r"""
## 5. Ajuste de hiperparámetros sobre **dev**

Búsqueda pequeña y honesta: 2 valores de dropout × 2 learning rates, 3 épocas cada una,
sobre el 20 % de train. Se elige por `val_accuracy` (dev), nunca por test.
"""
)

code(
    r"""
# ============================================================
# 5. Búsqueda de hiperparámetros (solo dev)
# ============================================================
from itertools import product


def build_vgg16_model(dropout=0.4, l2_reg=1e-4, lr=1e-3, base=None, name="vgg16"):
    if base is None:
        base = keras.applications.VGG16(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="entrada")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg), name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout, name="drop1")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg), name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(dropout, name="drop2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="salida")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def run_tuning():
    grid = list(product([0.3, 0.5], [1e-3, 3e-4]))
    steps = max(1, int(TUNING_FRAC * len(train_vgg)))
    tune_train = train_vgg.take(steps)

    rows = []
    for dropout, lr in grid:
        tag = f"tune_d{dropout}_lr{lr}"
        print(f"\n>>> Configuración: dropout={dropout} | lr={lr}")
        keras.utils.set_random_seed(SEED)
        model = build_vgg16_model(dropout=dropout, lr=lr, name=tag)
        hist = model.fit(
            tune_train, validation_data=dev_vgg,
            epochs=EPOCHS_TUNING, verbose=2,
        )
        best_dev = max(hist.history["val_accuracy"])
        rows.append({"dropout": dropout, "lr": lr, "val_accuracy": best_dev})
        del model
        keras.backend.clear_session()

    table = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    display(table)
    best = table.iloc[0]
    print(f"Mejor configuración: dropout={best.dropout} | lr={best.lr} "
          f"| val_accuracy={best.val_accuracy:.4f}")
    return float(best.dropout), float(best.lr), table


if RUN_TUNING:
    BEST_DROPOUT, BEST_LR, TUNING_TABLE = run_tuning()
else:
    BEST_DROPOUT, BEST_LR = 0.4, 1e-3
    print("Búsqueda omitida; se usan valores por defecto.")
"""
)

# ---------------------------------------------------------------------------
# 6. VGG16
# ---------------------------------------------------------------------------
md(
    r"""
## 6. Modelo A — Transfer learning con VGG16

VGG16 preentrenado en ImageNet. Congelamos **toda** la base convolucional y entrenamos
una cabeza nueva (GAP → 512 → 256 → softmax) con dropout, L2 y label smoothing.
"""
)

code(
    r"""
# ============================================================
# 6. Modelo A: VGG16 congelado + cabeza propia
# ============================================================
keras.backend.clear_session()
keras.utils.set_random_seed(SEED)

vgg_base = keras.applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
vgg_base.trainable = False

vgg_model = build_vgg16_model(
    dropout=BEST_DROPOUT, lr=BEST_LR, base=vgg_base, name="A_vgg16_frozen"
)
vgg_model.summary()

t0 = time.time()
vgg_hist = vgg_model.fit(
    train_vgg,
    validation_data=dev_vgg,
    epochs=EPOCHS_VGG,
    callbacks=make_callbacks("A_vgg16_frozen")[0],
    verbose=2,
)
vgg_train_time = time.time() - t0
print(f"Entrenamiento: {vgg_train_time/60:.1f} min")

plot_history(vgg_hist, "A · VGG16 congelado")
vgg_result = evaluate_model(vgg_model, test_vgg, "A · VGG16 (base congelada)")
vgg_result["train_time_min"] = round(vgg_train_time / 60, 1)

bias_variance_report(vgg_model, "A · VGG16", train_eval_vgg, dev_vgg)
"""
)

# ---------------------------------------------------------------------------
# 7. Fine-tuning
# ---------------------------------------------------------------------------
md(
    r"""
## 7. Modelo A+ — Fine-tuning del último bloque

Descongelamos `block5` y reentrenamos con `lr = 1e-5`.
"""
)

code(
    r"""
# ============================================================
# 7. Fine-tuning de block5
# ============================================================
vgg_base.trainable = True
for layer in vgg_base.layers:
    layer.trainable = layer.name.startswith("block5")

print("Capas entrenables:", sum(1 for l in vgg_model.layers if l.trainable))

vgg_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

t0 = time.time()
ft_hist = vgg_model.fit(
    train_vgg,
    validation_data=dev_vgg,
    epochs=EPOCHS_FT,
    callbacks=make_callbacks("Aplus_vgg16_finetuned")[0],
    verbose=2,
)
ft_train_time = time.time() - t0
print(f"Fine-tuning: {ft_train_time/60:.1f} min")

plot_history(ft_hist, "A+ · VGG16 fine-tuning block5")
ft_result = evaluate_model(vgg_model, test_vgg, "A+ · VGG16 + fine-tuning")
ft_result["train_time_min"] = round(ft_train_time / 60, 1)

bias_variance_report(vgg_model, "A+ · VGG16 fine-tuning", train_eval_vgg, dev_vgg)
"""
)

# ---------------------------------------------------------------------------
# 8. ResNet
# ---------------------------------------------------------------------------
md(
    r"""
## 8. Modelo B — Transfer learning con ResNet50V2

Segunda arquitectura preentrenada. Misma receta (base congelada + cabeza + augmentation).
"""
)

code(
    r"""
# ============================================================
# 8. Modelo B: ResNet50V2 congelada + cabeza propia
# ============================================================
keras.backend.clear_session()
keras.utils.set_random_seed(SEED)

res_base = keras.applications.ResNet50V2(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
res_base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="entrada")
x = res_base(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
x = layers.BatchNormalization(name="bn1")(x)
x = layers.Dropout(BEST_DROPOUT, name="drop1")(x)
x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
x = layers.BatchNormalization(name="bn2")(x)
x = layers.Dropout(BEST_DROPOUT, name="drop2")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="salida")(x)

res_model = keras.Model(inputs, outputs, name="B_resnet50v2")
res_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=BEST_LR),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

t0 = time.time()
res_hist = res_model.fit(
    train_res,
    validation_data=dev_res,
    epochs=EPOCHS_RESNET,
    callbacks=make_callbacks("B_resnet50v2")[0],
    verbose=2,
)
res_train_time = time.time() - t0
print(f"Entrenamiento: {res_train_time/60:.1f} min")

plot_history(res_hist, "B · ResNet50V2 congelada")
res_result = evaluate_model(res_model, test_res, "B · ResNet50V2")
res_result["train_time_min"] = round(res_train_time / 60, 1)

bias_variance_report(res_model, "B · ResNet50V2", train_eval_res, dev_res)
"""
)

# ---------------------------------------------------------------------------
# 9. CNN desde cero
# ---------------------------------------------------------------------------
md(
    r"""
## 9. Modelo C — CNN desde cero

Red propia sin pesos preentrenados: cuatro bloques `Conv-BN-ReLU ×2 + MaxPool`.
"""
)

code(
    r"""
# ============================================================
# 9. Modelo C: CNN desde cero
# ============================================================
def build_cnn(dropout=0.4, l2_reg=1e-4, lr=1e-3):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="entrada")

    x = inputs
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}a")(x)
        x = layers.BatchNormalization(name=f"bn{i}a")(x)
        x = layers.Activation("relu", name=f"relu{i}a")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}b")(x)
        x = layers.BatchNormalization(name=f"bn{i}b")(x)
        x = layers.Activation("relu", name=f"relu{i}b")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
        x = layers.Dropout(dropout * 0.5, name=f"drop{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg), name="fc1")(x)
    x = layers.Dropout(dropout, name="drop_fc")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="salida")(x)

    model = keras.Model(inputs, outputs, name="C_cnn_scratch")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


keras.backend.clear_session()
keras.utils.set_random_seed(SEED)

cnn_model = build_cnn(dropout=BEST_DROPOUT, lr=BEST_LR)
cnn_model.summary()

t0 = time.time()
cnn_hist = cnn_model.fit(
    train_scr,
    validation_data=dev_scr,
    epochs=EPOCHS_CNN,
    callbacks=make_callbacks("C_cnn_scratch", patience=3)[0],
    verbose=2,
)
cnn_train_time = time.time() - t0
print(f"Entrenamiento: {cnn_train_time/60:.1f} min")

plot_history(cnn_hist, "C · CNN desde cero")
cnn_result = evaluate_model(cnn_model, test_scr, "C · CNN desde cero")
cnn_result["train_time_min"] = round(cnn_train_time / 60, 1)

bias_variance_report(cnn_model, "C · CNN desde cero", train_eval_scr, dev_scr)
"""
)

# ---------------------------------------------------------------------------
# 10. Learning curve
# ---------------------------------------------------------------------------
md(
    r"""
## 10. Curvas de aprendizaje: sesgo vs. varianza

Reentrenamos la cabeza de VGG16 (base congelada) con fracciones
10 %, 25 %, 50 % y 100 % de train, 3 épocas cada vez.
"""
)

code(
    r"""
# ============================================================
# 10. Curvas de aprendizaje
# ============================================================
def learning_curve(fractions=(0.10, 0.25, 0.50, 1.00), epochs=3):
    rows = []
    total = len(train_vgg)
    for frac in fractions:
        steps = max(1, int(total * frac))
        print(f"\n>>> Fracción {frac:.0%} ({steps} lotes)")
        keras.utils.set_random_seed(SEED)
        model = build_vgg16_model(dropout=BEST_DROPOUT, lr=BEST_LR,
                                  name=f"lc_{int(frac*100)}")
        hist = model.fit(train_vgg.take(steps), validation_data=dev_vgg,
                         epochs=epochs, verbose=0)
        tr_acc = hist.history["accuracy"][-1]
        dv_acc = hist.history["val_accuracy"][-1]
        rows.append({"fraccion": frac, "n_lotes": steps,
                     "train_acc": tr_acc, "dev_acc": dv_acc,
                     "train_err": 1 - tr_acc, "dev_err": 1 - dv_acc})
        del model
        keras.backend.clear_session()
    return pd.DataFrame(rows)


if RUN_LEARNING_CURVE:
    lc = learning_curve()
    display(lc)

    plt.figure(figsize=(7, 4))
    plt.plot(lc["fraccion"], lc["train_err"], "o-", label="error train")
    plt.plot(lc["fraccion"], lc["dev_err"], "s-", label="error dev")
    plt.xlabel("Fracción de train")
    plt.ylabel("Error")
    plt.title("Curva de aprendizaje — VGG16 (base congelada)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
else:
    print("Curva de aprendizaje omitida.")
"""
)

# ---------------------------------------------------------------------------
# 11. Comparación
# ---------------------------------------------------------------------------
md(
    r"""
## 11. Evaluación final en test y comparación

### 11.0 Métrica única de evaluación (Ng caps. 8-9)

**Métrica a optimizar (optimizing):** F1 macro.
**Métrica satisfactoria (satisficing):** accuracy ≥ 0.90 y top-3 ≥ 0.99.
"""
)

code(
    r"""
# ============================================================
# 11. Tabla comparativa
# ============================================================
metrics_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                "f1_weighted", "top3_accuracy"]
comparison = pd.DataFrame(
    [{"modelo": r["modelo"], **{c: r[c] for c in metrics_cols},
      "tiempo_train_min": r.get("train_time_min", np.nan)} for r in RESULTS.values()]
).sort_values("accuracy", ascending=False).reset_index(drop=True)

display(comparison.style.format({
    "accuracy": "{:.4f}", "precision_macro": "{:.4f}", "recall_macro": "{:.4f}",
    "f1_macro": "{:.4f}", "f1_weighted": "{:.4f}", "top3_accuracy": "{:.4f}",
    "tiempo_train_min": "{:.1f}",
}))

comparison.to_csv(OUT_DIR / "comparacion_modelos.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4))
comparison.set_index("modelo")[["accuracy", "f1_macro"]].plot(
    kind="bar", ax=ax, color=["#4C72B0", "#DD8452"]
)
ax.set_ylim(0, 1.02)
ax.set_title("Comparación de modelos en test")
ax.set_ylabel("Métrica")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
"""
)

code(
    r"""
# ============================================================
# 11.1 Matrices de confusión (test)
# ============================================================
def plot_confusion(result, ax, title):
    cm = confusion_matrix(result["y_true"], result["y_pred"], normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)


n_models = len(RESULTS)
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
axes = axes.ravel()
for ax, result in zip(axes, RESULTS.values()):
    plot_confusion(result, ax, result["modelo"])
for ax in axes[n_models:]:
    ax.axis("off")
plt.tight_layout(); plt.show()

best_name = comparison.iloc[0]["modelo"]
best = RESULTS[best_name]
print(f"\nReporte por clase — {best_name}\n")
print(classification_report(best["y_true"], best["y_pred"],
                            target_names=CLASS_NAMES, digits=4, zero_division=0))
"""
)

# ---------------------------------------------------------------------------
# 12. Error analysis
# ---------------------------------------------------------------------------
md(
    r"""
## 12. Análisis de errores

### 12.0 Análisis de errores sobre **dev** (priorización)

Siguiendo a Ng (caps. 14-19), el análisis de errores se hace sobre el **dev set** para
priorizar qué arreglar; el test set se reserva para la estimación final sin sesgo.

* **Error óptimo (human-level)** ≈ 0 % para estas piezas.
* **Avoidable bias** = error_train − error_óptimo; **variance** = error_dev − error_train.
* **Eyeball dev set vs Blackbox dev set** (caps. 17-18).
* **Ceiling (techo) de cada categoría** (cap. 14).
* **Análisis de errores en train** (cap. 26) para modelos con sesgo alto.
"""
)

code(
    r"""
# ============================================================
# 12.0 Análisis de errores en dev (priorización)
# ============================================================
model_map = {"A · VGG16 (base congelada)": vgg_model,
             "A+ · VGG16 + fine-tuning": vgg_model,
             "B · ResNet50V2": res_model,
             "C · CNN desde cero": cnn_model}
ds_map = {"A · VGG16 (base congelada)": dev_vgg,
          "A+ · VGG16 + fine-tuning": dev_vgg,
          "B · ResNet50V2": dev_res,
          "C · CNN desde cero": dev_scr}

best_dev_result = evaluate_model(model_map[best_name], ds_map[best_name],
                                f"{best_name} (dev)", store=False)

dev_cm = confusion_matrix(best_dev_result["y_true"], best_dev_result["y_pred"])
dev_total_errors = int((best_dev_result["y_pred"] != best_dev_result["y_true"]).sum())
dev_rows = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and dev_cm[i, j] > 0:
            dev_rows.append({
                "real": CLASS_NAMES[i],
                "predicho": CLASS_NAMES[j],
                "n": int(dev_cm[i, j]),
                "ceiling_pct": 100 * dev_cm[i, j] / max(1, dev_total_errors),
            })
dev_pairs = pd.DataFrame(dev_rows).sort_values("n", ascending=False).reset_index(drop=True)
print(f"\nErrores totales en dev: {dev_total_errors}")
if len(dev_pairs) > 0:
    print(f"Ceiling si se arregla el par #1: {dev_pairs.iloc[0]['ceiling_pct']:.1f}% del error")
display(dev_pairs.head(10))

# Error por clase en train (Ng cap. 26)
train_ds_map = {"A · VGG16 (base congelada)": train_eval_vgg,
                "A+ · VGG16 + fine-tuning": train_eval_vgg,
                "B · ResNet50V2": train_eval_res,
                "C · CNN desde cero": train_eval_scr}
train_res_eval = evaluate_model(model_map[best_name], train_ds_map[best_name],
                               f"{best_name} (train)", store=False)
train_cm = confusion_matrix(train_res_eval["y_true"], train_res_eval["y_pred"], normalize="true")
train_per_class = pd.DataFrame({
    "clase": CLASS_NAMES,
    "train_accuracy": train_cm.diagonal(),
}).sort_values("train_accuracy")
print("\nError por clase en train (Ng cap. 26):")
display(train_per_class)

print("\nEyeball/Blackbox dev sets (Ng caps. 17-18):")
print(f"  dev total       : {len(best_dev_result['y_true'])} ejemplos")
print(f"  errores en dev  : {dev_total_errors}")
print(f"  Eyeball sugerido: ~2000 ejemplos")
print(f"  Blackbox sugerido: resto")
"""
)

code(
    r"""
# ============================================================
# 12. Análisis de errores del mejor modelo (test)
# ============================================================
def confusion_pairs(result, top=8):
    cm = confusion_matrix(result["y_true"], result["y_pred"])
    rows = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                rows.append({"real": CLASS_NAMES[i], "predicho": CLASS_NAMES[j],
                             "n": int(cm[i, j])})
    return pd.DataFrame(rows).sort_values("n", ascending=False).head(top).reset_index(drop=True)


pairs = confusion_pairs(best)
print(f"Pares más confundidos — {best_name}")
display(pairs)

cm = confusion_matrix(best["y_true"], best["y_pred"], normalize="true")
per_class = pd.DataFrame({
    "clase": CLASS_NAMES,
    "accuracy": cm.diagonal(),
    "soporte": np.bincount(best["y_true"], minlength=NUM_CLASSES),
}).sort_values("accuracy")

plt.figure(figsize=(9, 4))
sns.barplot(data=per_class, x="accuracy", y="clase", hue="clase", palette="viridis", legend=False)
plt.xlim(0, 1.05)
plt.title(f"Accuracy por clase — {best_name}")
plt.tight_layout(); plt.show()
display(per_class)
"""
)

code(
    r"""
# ============================================================
# 12.1 Galería de errores y confianza
# ============================================================
def show_errors(result, paths, n=12, correct=False):
    probs = result["y_prob"]
    conf = probs.max(axis=1)
    if correct:
        idx = np.where(result["y_pred"] == result["y_true"])[0]
        idx = idx[np.argsort(-conf[idx])]
        title = "Aciertos con mayor confianza"
    else:
        idx = np.where(result["y_pred"] != result["y_true"])[0]
        idx = idx[np.argsort(-conf[idx])]
        title = "Errores con mayor confianza"

    idx = idx[:n]
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows))
    axes = np.atleast_2d(axes).ravel()

    for ax, i in zip(axes, idx):
        if i < len(paths):
            ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.axis("off")
        ax.set_title(
            f"real: {CLASS_NAMES[result['y_true'][i]]}\n"
            f"pred: {CLASS_NAMES[result['y_pred'][i]]} ({conf[i]:.2f})",
            fontsize=7,
        )
    for ax in axes[len(idx):]:
        ax.axis("off")

    plt.suptitle(title, y=1.002)
    plt.tight_layout(); plt.show()
    return idx


print("Errores más confiados:")
_ = show_errors(best, TEST_PATHS, n=12, correct=False)

conf = best["y_prob"].max(axis=1)
is_correct = best["y_pred"] == best["y_true"]
plt.figure(figsize=(8, 4))
plt.hist(conf[is_correct], bins=40, alpha=0.65, label="aciertos", color="#55A868")
plt.hist(conf[~is_correct], bins=40, alpha=0.85, label="errores", color="#C44E52")
plt.xlabel("Confianza de la predicción")
plt.ylabel("N.º de imágenes")
plt.title(f"Distribución de confianza — {best_name}")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"Confianza media en aciertos: {conf[is_correct].mean():.4f}")
print(f"Confianza media en errores : {conf[~is_correct].mean():.4f}")
print(f"Errores con confianza > 0.9: {(conf[~is_correct] > 0.9).sum()}")
"""
)

# ---------------------------------------------------------------------------
# 13. Conclusiones
# ---------------------------------------------------------------------------
md(
    r"""
## 13. Conclusiones y limitaciones

### Qué se espera observar

* **Transfer learning > desde cero** en accuracy y eficiencia de datos.
* **El fine-tuning** suele mejorar la precisión en clases visualmente parecidas.
* **La CNN desde cero** necesita más épocas y tiende a sobreajustar antes.

### Lectura de sesgo y varianza

* Si el error de train es bajo y la brecha train-dev es grande → **varianza alta**.
* Si train y dev son altos y cercanos → **sesgo alto**.

### Playbook de Ng: qué hacer según el diagnóstico

| Diagnóstico (Ng caps. 23, 25, 27) | Acción concreta |
|---|---|
| Avoidable bias alto | Aumentar capacidad, reducir regularización, mejorar features, cambiar arquitectura. |
| Variance alta | Más datos, regularización, early stopping, augmentation más agresiva. |
| Ambos altos | Aumentar capacidad y regularización a la vez; revisar data mismatch. |
| Cerca del óptimo | Ensembling o más datos reales de cámara. |
| Data mismatch (train ≠ cámara real) | Redefinir dev/test con imágenes reales; ponderar datos o síntesis artificial. |

### Limitaciones honestas

1. **Dataset sintético**: las imágenes provienen de renderizados CAD.
2. **Clases semánticamente cercanas**: algunas piezas comparten geometría.
3. **Presupuesto de cómputo**: se limita el número de épocas.
4. **Una sola semilla**: no se reporta media ± desviación.

### Cuál conviene y por qué

| Criterio | Ganador esperado |
|---|---|
| Mejor accuracy absoluta | VGG16 + fine-tuning o ResNet50V2 |
| Mejor relación accuracy / costo | ResNet50V2 |
| Más simple de entrenar | VGG16 congelado |
| Menor latencia de inferencia | CNN desde cero |

**Conclusión operativa:** para producción industrial, partir de ResNet50V2 o VGG16 +
fine-tuning, con augmentation agresiva y validación con datos reales.
"""
)

code(
    r"""
# ============================================================
# 14. Guardado de artefactos
# ============================================================
import datetime as _dt

summary = {
    "fecha": _dt.datetime.now().isoformat(timespec="seconds"),
    "dataset": "beschue/industrial-classification-data-set",
    "clases": CLASS_NAMES,
    "comparacion": comparison.to_dict(orient="records"),
    "mejor_modelo": best_name,
    "tuning": TUNING_TABLE.to_dict(orient="records") if RUN_TUNING else None,
    "learning_curve": lc.to_dict(orient="records") if RUN_LEARNING_CURVE else None,
}

def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)

with open(OUT_DIR / "resumen_resultados.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2, default=_json_default)

print("Artefactos guardados en:", OUT_DIR)
for p in sorted(OUT_DIR.glob("*")):
    print("  -", p.name, f"({p.stat().st_size/1e6:.1f} MB)")
"""
)

code(
    r"""
# ============================================================
# 14.1 Descarga del resumen (solo Colab web)
# ============================================================
try:
    from google.colab import files as colab_files
    zip_path = Path("/content/resultados_parcial2.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_DIR / "comparacion_modelos.csv", "comparacion_modelos.csv")
        zf.write(OUT_DIR / "resumen_resultados.json", "resumen_resultados.json")
        for p in sorted(OUT_DIR.glob("*.keras")):
            zf.write(p, p.name)
    colab_files.download(str(zip_path))
except Exception as exc:
    print("Descarga omitida (no estás en Colab web):", exc)
"""
)

# ---------------------------------------------------------------------------
# Ensamblado del notebook
# ---------------------------------------------------------------------------
nb = {
    "cells": CELLS,
    "metadata": {
        "colab": {
            "name": "Parcial2_Deep_Learning.ipynb",
            "provenance": [],
            "collapsed_sections": [],
            "toc_visible": True,
            "gpuType": "T4",
        },
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook generado: {OUT}")
print(f"Celdas: {len(CELLS)} (código: {sum(1 for c in CELLS if c['cell_type']=='code')})")
