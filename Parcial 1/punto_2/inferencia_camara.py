import os
import cv2
import numpy as np
import kagglehub
from sklearn.preprocessing import LabelEncoder

# Configurar el backend antes de importar keras
os.environ["KERAS_BACKEND"] = "torch"
import keras

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "modelo_punto2.keras")

print(f"Cargando modelo desde {MODEL_PATH}...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print("Error al cargar el modelo. ¿Ya se terminó de entrenar?")
    print(e)
    exit()

print("Recuperando las clases originales para la interfaz...")
path = kagglehub.dataset_download("ardamavi/27-class-sign-language-dataset")
y_raw = np.load(os.path.join(path, "Y.npy"))
le = LabelEncoder()
le.fit(y_raw.flatten())
clases = le.classes_
print(f"Clases detectadas: {clases}")

IMG_SIZE = 64

# Inicializar cámara
cap = None
for cam_idx in [0, 1, 2]:
    # Intentar con DirectShow (recomendado en Windows)
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Cámara web (índice {cam_idx}) iniciada correctamente con DirectShow.")
        break
    
    # Intentar con el backend por defecto
    cap = cv2.VideoCapture(cam_idx)
    if cap.isOpened():
        print(f"Cámara web (índice {cam_idx}) iniciada correctamente.")
        break

if cap is None or not cap.isOpened():
    print("Error: No se pudo acceder a ninguna cámara web. Revisa los permisos o si otra app la está usando.")
    exit()

print("Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Hacer una copia para dibujar los resultados
    display_frame = frame.copy()
    
    # Definir una región de interés (ROI) en el centro de la pantalla
    # donde el usuario debe poner la mano
    height, width, _ = frame.shape
    roi_size = 300
    x1 = int(width/2 - roi_size/2)
    y1 = int(height/2 - roi_size/2)
    x2 = int(width/2 + roi_size/2)
    y2 = int(height/2 + roi_size/2)
    
    # Dibujar un rectángulo verde para la ROI
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display_frame, "Pon tu mano aqui", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Extraer la ROI para predecir
    roi = frame[y1:y2, x1:x2]
    
    # Si la ROI es válida
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        # Preprocesar igual que en el entrenamiento
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        roi_flat = roi_resized.flatten()
        
        # Normalizar y expandir dimensiones para el batch
        input_data = np.array([roi_flat], dtype='float32') / 255.0
        
        # Predecir
        pred_probs = model.predict(input_data, verbose=0)
        pred_class_idx = np.argmax(pred_probs, axis=1)[0]
        confidence = np.max(pred_probs)
        
        pred_label = clases[pred_class_idx]
        
        # Mostrar predicción si hay suficiente confianza
        if confidence > 0.5:
            text = f"Senal: {pred_label} ({confidence*100:.1f}%)"
            cv2.putText(display_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(display_frame, "Detectando...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar el frame resultante
    cv2.imshow("Reconocimiento de Lengua de Senas", display_frame)
    
    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
