import os
import io
import zipfile
import cv2
import h5py
import numpy as np

# Nombres anatómicos de los 5 dedos (Punto 3)
FINGER_NAMES = ['Pulgar', 'Índice', 'Medio', 'Anular', 'Meñique']
IMG_SIZE = 64
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH = os.path.join(SCRIPT_DIR, "modelo_punto3.keras")


def cargar_modelo():
    """Carga y evalúa dinámicamente cualquier arquitectura Dense+BN desde 'modelo_punto3.keras' (Modelo 2 o Modelo 3)."""
    if not os.path.exists(KERAS_PATH):
        print(f"Error: No se encontró el archivo del modelo en '{KERAS_PATH}'.")
        exit(1)

    try:
        with zipfile.ZipFile(KERAS_PATH, 'r') as z:
            weights_bytes = z.read('model.weights.h5')

        with h5py.File(io.BytesIO(weights_bytes), 'r') as f:
            layers = f['layers']
            dense_list = []
            for k in layers.keys():
                if 'dense' in k and 'vars' in layers[k] and '0' in layers[k]['vars']:
                    W = layers[k]['vars']['0'][()].astype(np.float32)
                    b = layers[k]['vars']['1'][()].astype(np.float32)
                    dense_list.append((W, b, k))

            bn_list = []
            for k in layers.keys():
                if 'batch_normalization' in k and 'vars' in layers[k] and '0' in layers[k]['vars']:
                    g = layers[k]['vars']['0'][()].astype(np.float32)
                    beta = layers[k]['vars']['1'][()].astype(np.float32)
                    m = layers[k]['vars']['2'][()].astype(np.float32)
                    v = layers[k]['vars']['3'][()].astype(np.float32)
                    bn_list.append((g, beta, m, v, k))

        stages = []
        curr_dim = 4096
        remaining_dense = dense_list.copy()

        while remaining_dense:
            found = False
            for i, (W, b, k) in enumerate(remaining_dense):
                if W.shape[0] == curr_dim:
                    found = True
                    out_dim = W.shape[1]
                    matching_bn = None
                    for bn in bn_list:
                        if bn[0].shape[0] == out_dim:
                            matching_bn = bn
                            break
                    stages.append((W, b, matching_bn))
                    curr_dim = out_dim
                    remaining_dense.pop(i)
                    break
            if not found:
                break

        def predecir_keras(x_input):
            h = x_input
            for i, (W, b, bn) in enumerate(stages):
                z = np.dot(h, W) + b
                if i == len(stages) - 1:
                    return (1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0))))[0]
                else:
                    h = np.maximum(0.0, z)
                    if bn is not None:
                        g, beta, m, v, _ = bn
                        h = g * (h - m) / np.sqrt(v + 1e-3) + beta
                        h = np.maximum(0.0, h)

        print(f"Modelo '.keras' cargado exitosamente ({len(stages)} capas) desde '{KERAS_PATH}'.")
        return predecir_keras
    except Exception as e:
        print(f"Error al leer '{KERAS_PATH}': {e}")
        exit(1)


# Cargar el modelo
predecir_dedos = cargar_modelo()


def ejecutar_camara():
    cap = None
    for cam_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            print(f"Cámara web (índice {cam_idx}) iniciada correctamente.")
            break

    if cap is None or not cap.isOpened():
        print("Error: No se pudo acceder a ninguna cámara web. Revisa los permisos o si otra app la está usando.")
        exit(1)

    umbral = 0.30

    print("Cámara iniciada. Coloca tu mano en el recuadro verde.")
    print("Controles: [+/-] Ajustar umbral de detección | [q] Salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Modo espejo para interacción natural
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        roi_size = 280
        x1 = int(width / 2 - roi_size / 2)
        y1 = int(height / 2 - roi_size / 2)
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        # Dibujar recuadro de la región de interés
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Pon tu mano aqui", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Extraer ROI
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 1. Análisis morfológico para descartar puño cerrado
            # En un puño, el objeto es compacto (solidity > 0.78) y de baja altura (bh < 170)
            _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            es_puno = False
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > 4000:
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 1.0
                    _, _, _, bh = cv2.boundingRect(c)
                    if solidity > 0.78 and bh < 170:
                        es_puno = True

            # 2. Inferencia con la red neuronal densa
            roi_norm = cv2.normalize(roi_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            roi_resized = cv2.resize(roi_norm, (IMG_SIZE, IMG_SIZE))
            input_data = np.array([roi_resized.flatten()], dtype='float32') / 255.0
            probs = predecir_dedos(input_data)

            y_pos = 35
            if es_puno:
                dedos_activos = []
                cv2.putText(frame, "Dedos detectados (0/5):", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(frame, "Puno cerrado / Ninguno", (10, y_pos + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                dedos_activos = [FINGER_NAMES[i] for i, p in enumerate(probs) if p >= umbral]
                cv2.putText(frame, f"Dedos detectados ({len(dedos_activos)}/5):", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                if dedos_activos:
                    texto = ", ".join(dedos_activos)
                    cv2.putText(frame, texto, (10, y_pos + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Mano cerrada / Ninguno", (10, y_pos + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Mostrar probabilidades individuales
            for i, (name, p) in enumerate(zip(FINGER_NAMES, probs)):
                p_display = 0.05 if es_puno else p
                color = (0, 255, 0) if (not es_puno and p >= umbral) else (180, 180, 180)
                cv2.putText(frame, f"{name}: {p_display*100:.0f}%", (10, y_pos + 70 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            cv2.putText(frame, f"Umbral: {int(umbral*100)}% (+/- para calibrar)",
                        (10, y_pos + 70 + 5 * 22 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow("Punto 3 - Detector de Dedos Extendidos", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('+'), ord('='), ord('w')]:
            umbral = min(0.90, umbral + 0.05)
            print(f"Umbral ajustado a: {int(umbral*100)}%")
        elif key in [ord('-'), ord('_'), ord('s')]:
            umbral = max(0.10, umbral - 0.05)
            print(f"Umbral ajustado a: {int(umbral*100)}%")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ejecutar_camara()
