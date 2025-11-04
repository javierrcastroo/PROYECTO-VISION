import cv2
import os

# carpeta donde vamos a guardar las fotos
SAVE_DIR = "calib_images"
os.makedirs(SAVE_DIR, exist_ok=True)
   
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

print("[INFO] Pulsa ESPACIO para guardar una foto del tablero.")
print("[INFO] Pulsa q para salir.")

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Captura calibracion", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # espacio
        filename = os.path.join(SAVE_DIR, f"calib_{counter:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[INFO] Guardada {filename}")
        counter += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
