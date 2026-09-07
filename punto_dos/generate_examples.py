import os
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.preprocessing import LabelEncoder
import cv2

# Descargar/obtener ruta del dataset
path = kagglehub.dataset_download("ardamavi/27-class-sign-language-dataset")
X_raw = np.load(os.path.join(path, "X.npy"))
y_raw = np.load(os.path.join(path, "Y.npy")).flatten()

# Obtener clases únicas y encoder
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
clases = le.classes_

# Encontrar un ejemplo para cada clase
examples = {}
for i, label in enumerate(clases):
    # Encontrar el primer indice donde y_raw == label
    idx = np.where(y_raw == label)[0][0]
    examples[label] = X_raw[idx]

# Crear carpeta ejemplo si no existe
out_dir = "ejemplo"
os.makedirs(out_dir, exist_ok=True)

for i, label in enumerate(clases):
    # Guardar cada imagen individualmente
    img = examples[label]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convertir a BGR para guardar con cv2
    file_path = os.path.join(out_dir, f"sena_{label}.png")
    cv2.imwrite(file_path, img_bgr)

print(f"27 imágenes guardadas en la carpeta: {out_dir}")
