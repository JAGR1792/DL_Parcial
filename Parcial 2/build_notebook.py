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

**Curso:** Deep Learning (CC-IA-04)
**Integrantes:** J. A. García Romero · S. J. Rodríguez Gutiérrez · C. A. Díaz García

**Dataset:** [Industrial Dataset](https://www.kaggle.com/datasets/beschue/industrial-classification-data-set)
(MVTec ITODD renderizado con SynthiCAD). 10 clases de piezas industriales,
100 000 imágenes RGB de 224×224, repartidas en 70 000 train / 10 000 valid / 20 000 test.

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
las rutas de trabajo. Los artefactos se guardan en Drive para sobrevivir a una desconexión.
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

# Precisión mixta: acelera el entrenamiento en GPU sin cambiar el resultado numérico
# de forma apreciable. La capa final de cada modelo se declara en float32.
keras.mixed_precision.set_global_policy("mixed_float16")

print("Python     :", sys.version.split()[0])
print("TensorFlow :", tf.__version__)
print("Keras      :", keras.__version__)
print("Política   :", keras.mixed_precision.global_policy().name)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPU        :", tf.config.experimental.get_device_details(gpus[0]).get("device_name"))
else:
    print("GPU        : NINGUNA -- el entrenamiento será lento (Runtime > Change runtime type > T4 GPU)")
"""
)

code(
    r"""
# ============================================================
# Parámetros globales del experimento
# ============================================================
IMG_SIZE   = 224          # resolución nativa del dataset
BATCH_SIZE = 64
NUM_CLASSES = 10

EPOCHS_TUNING = 3         # épocas por configuración en la búsqueda de hiperparámetros
EPOCHS_VGG    = 10        # cabeza sobre VGG16 congelado
EPOCHS_FT     = 3         # fine-tuning del último bloque de VGG16
EPOCHS_RESNET = 10
EPOCHS_CNN    = 15        # CNN desde cero

TUNING_FRAC = 0.20        # fracción de train usada en la búsqueda de hiperparámetros
RUN_TUNING  = True        # poner False para saltar la búsqueda
RUN_LEARNING_CURVE = True # curvas de aprendizaje (sesgo/varianza)

FAST_DEV_RUN = False      # True = corre todo con un subconjunto mínimo (para probar el pipeline)
FAST_FRAC    = 0.05

USE_DRIVE = True          # guardar checkpoints y resultados en Google Drive

AUTOTUNE = tf.data.AUTOTUNE

print(f"IMG_SIZE={IMG_SIZE} | BATCH_SIZE={BATCH_SIZE} | mixed_float16 | seed={SEED}")
print("FAST_DEV_RUN =", FAST_DEV_RUN)
"""
)

code(
    r"""
# ============================================================
# Directorio de salida (Drive si está disponible)
# ============================================================
OUT_DIR = Path("/content/out")

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT_DIR = Path("/content/drive/MyDrive/Parcial2_DL")
        print("Drive montado.")
    except Exception as exc:
        print("No se pudo montar Drive (¿no estás en Colab?):", exc)

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

El dataset vive en Kaggle y requiere autenticación. La celda resuelve las credenciales en
este orden:

1. **Secretos de Colab** (`KAGGLE_USERNAME` y `KAGGLE_KEY`): la forma recomendada, porque
   las claves nunca quedan escritas en el notebook.
2. Archivo `~/.kaggle/kaggle.json` si ya existe en el runtime.
3. Diálogo de subida (solo en la interfaz web de Colab).

Con las credenciales listas descarga con `kagglehub` y, si falla, cae al CLI de Kaggle;
después detecta automáticamente las carpetas `train` / `valid` / `test`.

> **Cómo obtener las credenciales:** Kaggle → foto de perfil → *Settings* → sección *API* →
> *Create New Token*. Se descarga `kaggle.json`, que contiene `username` y `key`.
> En Colab se guardan como secretos con el botón de la llave del panel izquierdo
> (nombres exactos: `KAGGLE_USERNAME` y `KAGGLE_KEY`) y se activa el acceso para este notebook.
"""
)

code(
    r"""
# ============================================================
# 1. Descarga y localización del dataset
# ============================================================
KAGGLE_DATASET = "beschue/industrial-classification-data-set"
DATA_ROOT = Path("/content/data")


def ensure_kaggle_credentials() -> bool:
    # Orden de preferencia:
    #   1) Secretos de Colab (KAGGLE_USERNAME / KAGGLE_KEY) -> nunca quedan en el notebook
    #   2) archivo ~/.kaggle/kaggle.json
    #   3) diálogo de subida (solo funciona en la interfaz web de Colab)
    try:
        from google.colab import userdata
        for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            value = userdata.get(key)
            if value:
                os.environ[key] = value
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            print("Credenciales de Kaggle tomadas de los Secretos de Colab.")
            return True
    except Exception as exc:
        print("Secretos de Colab no disponibles:", exc)

    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists():
        print("Credenciales de Kaggle encontradas en", cred)
        return True

    print("No hay kaggle.json. Abriendo diálogo de subida...")
    try:
        from google.colab import files
        uploaded = files.upload()
        if "kaggle.json" in uploaded:
            cred.parent.mkdir(parents=True, exist_ok=True)
            cred.write_bytes(uploaded["kaggle.json"])
            os.chmod(cred, 0o600)
            print("kaggle.json guardado en", cred)
            return True
        print("No se subió kaggle.json.")
    except Exception as exc:
        print("No se pudo subir kaggle.json:", exc)
    return False


def find_split_dirs(root: Path) -> dict:
    # Busca carpetas cuyo nombre sea train/valid/test en cualquier nivel.
    aliases = {
        "train": {"train", "training", "entrenamiento"},
        "valid": {"valid", "validation", "val", "dev", "validacion", "validación"},
        "test":  {"test", "testing", "prueba"},
    }
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            name = path.name.lower()
            for split, names in aliases.items():
                if name in names and split not in found:
                    found[split] = path
    return found


def download_dataset() -> None:
    if DATA_ROOT.exists() and (DATA_ROOT / "train").exists():
        print("El dataset ya está en", DATA_ROOT)
        return

    if not ensure_kaggle_credentials():
        raise RuntimeError("Sin credenciales de Kaggle no se puede descargar el dataset.")

    source = None
    try:
        import kagglehub
        source = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        print("kagglehub descargó en:", source)
    except Exception as exc:
        print("kagglehub falló, se intenta el CLI:", exc)
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", str(DATA_ROOT), "--unzip"],
            check=True,
        )
        source = DATA_ROOT

    splits = find_split_dirs(source)
    if not splits:
        raise RuntimeError(f"No se encontraron carpetas train/valid/test dentro de {source}")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for split, path in splits.items():
        dest = DATA_ROOT / split
        if dest.exists() or dest.is_symlink():
            continue
        try:
            dest.symlink_to(path, target_is_directory=True)
        except Exception:
            shutil.copytree(path, dest)
        print(f"  {split:5s} -> {path}")

    if (DATA_ROOT / "valid").exists() and not (DATA_ROOT / "dev").exists():
        (DATA_ROOT / "dev").symlink_to(DATA_ROOT / "valid", target_is_directory=True)


download_dataset()

# Estructura resultante (2 niveles)
for split in ["train", "valid", "test"]:
    folder = DATA_ROOT / split
    if folder.exists():
        classes = sorted([p.name for p in folder.iterdir() if p.is_dir()])
        print(f"{split:5s}: {len(classes)} clases -> {classes}")
"""
)

code(
    r"""
# ============================================================
# 1.1 Verificación de la estructura y conteo de imágenes
# ============================================================
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def resolve_class_root(split_dir: Path) -> Path:
    # Si la partición tiene una sola subcarpeta tipo "images", las clases están dentro.
    subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name.lower() in {"images", "image", "imagenes", "imágenes"}:
        return subdirs[0]
    return split_dir


def count_images(folder: Path) -> dict:
    counts = {}
    if not folder.exists():
        return counts
    folder = resolve_class_root(folder)
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        n = sum(1 for f in class_dir.rglob("*") if f.suffix.lower() in SUPPORTED)
        counts[class_dir.name] = n
    return counts


COUNTS = {split: count_images(DATA_ROOT / split) for split in ["train", "valid", "test"]}
for split, counts in COUNTS.items():
    print(f"{split:5s}: {sum(counts.values()):6d} imágenes en {len(counts)} clases")

CLASS_NAMES = sorted(COUNTS["train"].keys())
assert len(CLASS_NAMES) == NUM_CLASSES, f"Se esperaban {NUM_CLASSES} clases, hay {len(CLASS_NAMES)}"

# Comprobación de consistencia entre particiones
for split in ["valid", "test"]:
    assert sorted(COUNTS[split].keys()) == CLASS_NAMES, f"Clases distintas en {split}"

balance = pd.DataFrame(COUNTS).rename_axis("clase").reset_index()
balance["total"] = balance[["train", "valid", "test"]].sum(axis=1)
display(balance)
print("\nClases:", CLASS_NAMES)
"""
)

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
md(
    r"""
## 2. Análisis exploratorio

Antes de modelar conviene mirar los datos: distribución de clases, tamaños y ejemplos.
El dataset es sintético y balanceado (7 000 imágenes de train por clase), lo que elimina
la necesidad de *class weighting*, pero **no** elimina el riesgo de sobreajustar texturas.
"""
)

code(
    r"""
# ============================================================
# 2. Distribución de clases y ejemplos
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4))
balance.set_index("clase")[["train", "valid", "test"]].plot(
    kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52"], width=0.8
)
ax.set_title("Distribución de imágenes por clase y partición")
ax.set_ylabel("N.º de imágenes")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="Partición")
plt.tight_layout()
plt.show()

# Tamaños reales de las imágenes (muestra)
from PIL import Image
sample_paths = [p for p in (DATA_ROOT / "train").rglob("*") if p.suffix.lower() in SUPPORTED][:200]
sizes = Counter(Image.open(p).size for p in sample_paths)
print("Tamaños encontrados en una muestra de 200 imágenes:", dict(sizes))
"""
)

code(
    r"""
# ============================================================
# 2.1 Galería de ejemplos (3 por clase)
# ============================================================
fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(7, 2.2 * NUM_CLASSES))
for row, class_name in enumerate(CLASS_NAMES):
    class_dir = DATA_ROOT / "train" / class_name
    files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in SUPPORTED])[:3]
    for col in range(3):
        ax = axes[row, col]
        if col < len(files):
            ax.imshow(Image.open(files[col]).convert("RGB"))
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
| `train` | entrenamiento | 70 000 |
| `valid` | **dev** (decisiones, hiperparámetros) | 10 000 |
| `test` | **test** (evaluación final, una sola vez) | 20 000 |

**Decisiones del pipeline**

* `image_dataset_from_directory` sin batching → permite aumentar imagen por imagen.
* **Data augmentation** solo en train: volteo horizontal, recorte aleatorio (85–100 % del área)
  con reescalado, y jitter de brillo, contraste y saturación.
* Cada arquitectura usa su propio `preprocess_input` (VGG y ResNet lo necesitan).
* `prefetch` + `num_parallel_calls` para que la GPU no espere a la CPU.
"""
)

code(
    r"""
# ============================================================
# 3. Pipeline de datos
# ============================================================
def make_raw_dataset(directory, class_names, shuffle):
    # Dataset sin batchear: una imagen (224,224,3) en [0,255] y etiqueta one-hot.
    return keras.utils.image_dataset_from_directory(
        resolve_class_root(Path(directory)),
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=shuffle,
        seed=SEED if shuffle else None,
        interpolation="bilinear",
    )


def augment_single(image, label):
    # Augmentation geométrica + fotométrica sobre UNA imagen en [0,255].
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


def preprocess_identity(image, label):
    return image, label


def preprocess_vgg(image, label):
    return keras.applications.vgg16.preprocess_input(image), label


def preprocess_resnet(image, label):
    return keras.applications.resnet_v2.preprocess_input(image), label


def preprocess_scratch(image, label):
    return image / 255.0, label


def build_pipeline(directory, class_names, preprocess_fn, training=False, fast=False):
    ds = make_raw_dataset(directory, class_names, shuffle=training)
    if training:
        ds = ds.map(augment_single, num_parallel_calls=AUTOTUNE)
        if fast:
            ds = ds.take(int(FAST_FRAC * 70000))
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# Rutas de test (se conservan para el análisis de errores)
raw_test = make_raw_dataset(DATA_ROOT / "test", CLASS_NAMES, shuffle=False)
TEST_PATHS = list(raw_test.file_paths)

# Un pipeline por preprocesamiento
train_vgg = build_pipeline(DATA_ROOT / "train", CLASS_NAMES, preprocess_vgg, training=True, fast=FAST_DEV_RUN)
dev_vgg   = build_pipeline(DATA_ROOT / "valid", CLASS_NAMES, preprocess_vgg)
test_vgg  = raw_test.map(preprocess_vgg, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_res = build_pipeline(DATA_ROOT / "train", CLASS_NAMES, preprocess_resnet, training=True, fast=FAST_DEV_RUN)
dev_res   = build_pipeline(DATA_ROOT / "valid", CLASS_NAMES, preprocess_resnet)
test_res  = raw_test.map(preprocess_resnet, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_scr = build_pipeline(DATA_ROOT / "train", CLASS_NAMES, preprocess_scratch, training=True, fast=FAST_DEV_RUN)
dev_scr   = build_pipeline(DATA_ROOT / "valid", CLASS_NAMES, preprocess_scratch)
test_scr  = raw_test.map(preprocess_scratch, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Versiones SIN augmentation de train (para medir el error de train real, como en Ng)
train_eval_vgg = build_pipeline(DATA_ROOT / "train", CLASS_NAMES, preprocess_vgg, training=False, fast=FAST_DEV_RUN)
train_eval_res = build_pipeline(DATA_ROOT / "train", CLASS_NAMES, preprocess_resnet, training=False, fast=FAST_DEV_RUN)
train_eval_scr = build_pipeline(DATA_ROOT / "train", CLASS_NAMES, preprocess_scratch, training=False, fast=FAST_DEV_RUN)

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
    ds = make_raw_dataset(DATA_ROOT / "train", CLASS_NAMES, shuffle=True).take(n)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8))
    for i, (image, label) in enumerate(ds):
        axes[0, i].imshow(image.numpy().astype("uint8"))
        axes[0, i].set_title(CLASS_NAMES[int(tf.argmax(label))], fontsize=8)
        axes[0, i].axis("off")

        aug, _ = augment_single(tf.cast(image, tf.float32), label)
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
RESULTS = {}   # nombre -> dict con métricas y predicciones


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
    # Evalúa en un dataset y devuelve un dict con las métricas principales.
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


def bias_variance_report(model, name, train_ds, dev_ds, sample_train=10000):
    # Compara error de train (muestreado) con error de dev y emite un diagnóstico.
    train_sample = train_ds.unbatch().take(sample_train).batch(BATCH_SIZE)
    tr_loss, tr_acc = model.evaluate(train_sample, verbose=0)
    dv_loss, dv_acc = model.evaluate(dev_ds, verbose=0)

    tr_err, dv_err = 1 - tr_acc, 1 - dv_acc
    gap = dv_err - tr_err

    if tr_err > 0.10:
        verdict = "SESGO ALTO (bias): el modelo no ajusta ni siquiera el train."
        action = "Aumentar capacidad, entrenar más épocas o mejorar las features."
    elif gap > 0.05:
        verdict = "VARIANZA ALTA (overfitting): gran brecha train-dev."
        action = "Más regularización/augmentation o más datos."
    else:
        verdict = "BUEN AJUSTE: error de train bajo y brecha train-dev pequeña."
        action = "El cuello de botella está en el techo del modelo o en los datos."

    print(f"{name}: train_acc={tr_acc:.4f} | dev_acc={dv_acc:.4f} | "
          f"gap={gap:+.4f}")
    print("  ->", verdict)
    print("  ->", action)

    return {"modelo": name, "train_acc": tr_acc, "dev_acc": dv_acc,
            "train_err": tr_err, "dev_err": dv_err, "gap": gap, "diagnostico": verdict}
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

Esto es exactamente el ciclo de Ng: proponer configuración → medir en dev → quedarse con la mejor.
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

Con la base congelada, el modelo aprende a *reutilizar* los detectores de bordes, texturas y
formas de ImageNet; el ajuste fino del último bloque se hace después, con un learning rate bajo.
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

Descongelamos `block5` (las capas convolucionales de mayor nivel) y reentrenamos con
`lr = 1e-5`. Es el paso opcional que propone el enunciado: adaptar los filtros de alto nivel
a las texturas metálicas del dataset sin destruir lo aprendido en ImageNet.

Se recompila con un learning rate bajo; si no, el fine-tuning borra los pesos preentrenados.
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

Segunda arquitectura preentrenada. ResNet50V2 introduce conexiones residuales y
*pre-activation*, lo que facilita entrenar redes profundas y suele generalizar mejor que VGG
con menos parámetros efectivos.

Se mantiene la misma receta (base congelada + cabeza + augmentation + callbacks) para que la
comparación sean manzanas con manzanas.
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

Red propia sin pesos preentrenados: cuatro bloques `Conv-BN-ReLU ×2 + MaxPool`,
con dropout creciente y `GlobalAveragePooling`. Es el *baseline* honesto: si el transfer
learning no le gana claramente, hay que preguntarse si el dataset es demasiado fácil o
si las features de ImageNet no aportan.

Se entrena con la misma augmentation y los mismos callbacks.
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

Para diagnosticar de verdad hace falta ver cómo evolucionan los errores al crecer el conjunto
de entrenamiento. Reentrenamos la cabeza de VGG16 (base congelada) con fracciones
10 %, 25 %, 50 % y 100 % de train, 3 épocas cada vez.

Lectura esperada:

* **Curvas que convergen a un error alto** → sesgo alto (el modelo se queda corto).
* **Brecha grande y persistente entre train y dev** → varianza alta (hace falta más datos o regularización).
"""
)

code(
    r"""
# ============================================================
# 10. Curvas de aprendizaje
# ============================================================
def learning_curve(fractions=(0.10, 0.25, 0.50, 1.00), epochs=3):
    rows = []
    total_steps = len(train_vgg)
    for frac in fractions:
        steps = max(1, int(total_steps * frac))
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

Aquí —y solo aquí— se usa el conjunto de test. La tabla resume las métricas de los cuatro
modelos evaluados (tres estrategias + la variante con fine-tuning).
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

# Gráfico de barras comparativo
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

# Reporte por clase del mejor modelo
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

El análisis de errores de Ng: mirar los fallos uno por uno, agruparlos y decidir qué
haría más daño corregir. Aquí:

1. los pares de clases más confundidos;
2. una galería de imágenes mal clasificadas con su confianza;
3. la distribución de confianza (aciertos vs. errores);
4. la accuracy por clase, para detectar clases sistemáticamente débiles.
"""
)

code(
    r"""
# ============================================================
# 12. Análisis de errores del mejor modelo
# ============================================================
def confusion_pairs(result, top=8):
    cm = confusion_matrix(result["y_true"], result["y_pred"])
    rows = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "real": CLASS_NAMES[i],
                    "predicho": CLASS_NAMES[j],
                    "n": int(cm[i, j]),
                })
    return pd.DataFrame(rows).sort_values("n", ascending=False).head(top).reset_index(drop=True)


pairs = confusion_pairs(best)
print(f"Pares más confundidos — {best_name}")
display(pairs)

# Accuracy por clase
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
print("Aciertos más confiados:")
_ = show_errors(best, TEST_PATHS, n=8, correct=True)
"""
)

code(
    r"""
# ============================================================
# 12.2 Confianza: aciertos vs. errores
# ============================================================
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

* **Transfer learning > desde cero** en accuracy y, sobre todo, en eficiencia de datos:
  VGG16 y ResNet50V2 parten de filtros que ya detectan bordes, esquinas y texturas.
* **El fine-tuning** suele mejorar la precisión en las clases visualmente parecidas
  (por ejemplo *engine part cooler round* vs. *square*), porque adapta los filtros de alto
  nivel al acabado metálico sintético del dataset.
* **La CNN desde cero** necesita muchas más épocas y tiende a sobreajustar antes; su brecha
  train-dev suele ser la más grande. Aun así, es el control que demuestra el valor del
  preentrenamiento.

### Lectura de sesgo y varianza

* Si el error de train es bajo y la brecha train-dev es grande → **varianza alta**:
  más augmentación, dropout más agresivo o más datos.
* Si train y dev son altos y cercanos → **sesgo alto**: hace falta más capacidad o mejores
  features. La curva de aprendizaje permite distinguir ambos casos al mirar la forma.

### Limitaciones honestas

1. **Dataset sintético**: las imágenes provienen de renderizados CAD. Un clasificador con
   99 % aquí puede degradarse contra una cámara real (dominio distinto).
2. **Clases semánticamente cercanas**: algunas piezas comparten geometría y solo cambian
   detalles; la matriz de confusión lo evidencia.
3. **Presupuesto de cómputo**: se limita el número de épocas y la búsqueda de
   hiperparámetros; no se hace una búsqueda exhaustiva ni *ensembling*.
4. **Una sola semilla**: no se reporta media ± desviación sobre varias repeticiones.
   Es la mejora experimental más clara si hubiera más tiempo.

### Cuál conviene y por qué

| Criterio | Ganador esperado |
|---|---|
| Mejor accuracy absoluta | VGG16 + fine-tuning o ResNet50V2 |
| Mejor relación accuracy / costo | ResNet50V2 (menos parámetros que VGG16) |
| Más simple de entrenar | VGG16 congelado (la cabeza es minúscula) |
| Menor latencia de inferencia | CNN desde cero, si la accuracy alcanza |

**Conclusión operativa:** para producción industrial con imágenes nuevas de cámara, conviene
partir de ResNet50V2 o VGG16 + fine-tuning, con augmentation agresiva y validación cruzada
con datos reales; el modelo desde cero se justifica solo si hay restricciones de licencia
o de cómputo en inferencia.
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
    "dataset": KAGGLE_DATASET,
    "clases": CLASS_NAMES,
    "comparacion": comparison.to_dict(orient="records"),
    "mejor_modelo": best_name,
    "tuning": TUNING_TABLE.to_dict(orient="records") if RUN_TUNING else None,
    "learning_curve": lc.to_dict(orient="records") if RUN_LEARNING_CURVE else None,
}

def _json_default(obj):
    # Convierte tipos numpy a tipos nativos para que json.dump no falle.
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
# 14.1 Descarga del resumen (opcional)
# ============================================================
try:
    from google.colab import files as colab_files
    zip_path = Path("/content/resultados_parcial2.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_DIR / "comparacion_modelos.csv", "comparacion_modelos.csv")
        zf.write(OUT_DIR / "resumen_resultados.json", "resumen_resultados.json")
        # Solo los pesos del mejor modelo (los tres completos pesan demasiado)
        best_ckpt = OUT_DIR / f"{best_name.split(' ')[-1]}.keras"
        for p in sorted(OUT_DIR.glob("*.keras")):
            if p.name.startswith(best_name.split(" ")[0]) or p == best_ckpt:
                zf.write(p, p.name)
    colab_files.download(str(zip_path))
except Exception as exc:
    print("Descarga omitida:", exc)
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
