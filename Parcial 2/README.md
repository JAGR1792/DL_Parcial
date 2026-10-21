# Parcial 2 — Deep Learning

Clasificación de imágenes industriales comparando tres estrategias de modelado,
aplicando las prácticas del libro de Andrew Ng (Machine Learning Yearning).

## Archivos

| Archivo | Descripción |
|---|---|
| `Parcial2_Deep_Learning.ipynb` | Notebook principal (documentado, listo para correr) |
| `build_notebook.py` | Generador del notebook (editar esto y regenerar) |
| `Parcial2_Deep_Learning.pdf` / `.md` | Enunciado del parcial |
| `Machine_Learning_Yearning.md` | Libro de Andrew Ng en markdown |
| `requirements.txt` | Dependencias |

## Dataset

**Industrial Classification Data Set** (Kaggle): `beschue/industrial-classification-data-set`

- 10 clases de piezas industriales, 100 000 imágenes RGB de 224×224.
- 70 000 train / 10 000 valid / 20 000 test.
- **Estructura plana**: `Train_Dataset/images/adapter_plate_triangular000000.jpg`
  (la clase se extrae del nombre del archivo, no de subcarpetas).
- **No está en el repo** (1.2 GB). Se descarga solo o se monta en Kaggle.

## Puesta en marcha

### Opción A — Colab (recomendada)

1. Subir `Parcial2_Deep_Learning.ipynb` a [colab.research.google.com](https://colab.research.google.com/) → `File > Upload notebook`.
2. Panel izquierdo → ícono de la **llave (Secrets)** → agregar `KAGGLE_API_TOKEN` con tu token `KGAT_...` y activar "Notebook access".
3. `Runtime > Change runtime type > GPU` (T4 o L4).
4. `Runtime > Run all`.

### Opción B — Kaggle

1. Crear un notebook en Kaggle y agregar el dataset `beschue/industrial-classification-data-set` como input.
2. Subir `Parcial2_Deep_Learning.ipynb`.
3. Activar GPU (T4 x2 o P100) y `Run all`.

### Opción C — Local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

El notebook auto-detecta el entorno:

- Si existe `data/DataSet/` → usa el dataset local (crear con el zip descargado).
- Si existe `/kaggle/input/` → usa el dataset de Kaggle.
- Si no → lo descarga con `kagglehub` (requiere token).

> **Nota:** en Windows, TensorFlow ≥ 2.11 no usa GPU nativo. Para usar la GPU
> local, conviene WSL2 o Linux. En Colab/Kaggle la GPU se usa directo.

## Estructura del notebook

1. Configuración y entorno (auto-detección Kaggle/Colab/local).
2. Obtención del dataset (estructura plana, clase desde el nombre).
3. EDA: distribución de clases y galería.
4. Pipeline `tf.data` con data augmentation.
5. Utilidades de evaluación + reporte de sesgo/varianza (Ng caps. 20-27).
6. Ajuste de hiperparámetros sobre dev.
7. **Modelo A:** VGG16 congelado + cabeza propia.
8. **Modelo A+:** fine-tuning de `block5`.
9. **Modelo B:** ResNet50V2.
10. **Modelo C:** CNN desde cero.
11. Curvas de aprendizaje (sesgo vs. varianza).
12. Evaluación final en test y comparación.
13. Análisis de errores (pares confundidos, galería, confianza).
14. Conclusiones + guardado de artefactos.

## Prácticas de Ng aplicadas

- Partición train/dev/test de la misma distribución (caps. 5-12).
- Métrica única de evaluación + optimizing/satisficing (caps. 8-9).
- Avoidable bias / variance con error óptimo human-level (caps. 20-22).
- Error analysis con ceiling por categoría (caps. 14-15).
- Eyeball / Blackbox dev sets (caps. 17-18).
- Error analysis en train para sesgo alto (cap. 26).
- Playbook de acciones según diagnóstico (caps. 23, 25, 27).
- Learning curves train + dev vs tamaño (caps. 28-32).

## Interruptores (arriba del notebook)

| Variable | Efecto |
|---|---|
| `FAST_DEV_RUN = True` | Corre con un subconjunto pequeño (probar pipeline) |
| `RUN_TUNING = False` | Salta la búsqueda de hiperparámetros |
| `RUN_LEARNING_CURVE = False` | Salta las curvas de aprendizaje |
