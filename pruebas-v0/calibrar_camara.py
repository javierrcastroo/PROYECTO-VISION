import cv2
import numpy as np
import glob

# número de esquinas internas del tablero 
# si imprimiste un tablero típico de 9x6 (muy común),
# pon (9, 6). Si tu patrón es otro, cambia estos números.
CHESSBOARD_SIZE = (9, 6)

# criterios de refinado de esquinas
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# preparar puntos 3D: (0,0,0), (1,0,0), ..., en plano z=0
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

objpoints = []  # puntos 3D reales
imgpoints = []  # puntos 2D detectados

images = glob.glob("calib_images/*.jpg")
if len(images) == 0:
    raise RuntimeError("No hay imágenes en calib_images/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # buscar el patrón
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)

        # refinar esquinas detectadas
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            CRIT
        )
        imgpoints.append(corners2)

        # dibujar y mostrar (opcional)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Deteccion', img)
        cv2.waitKey(200)
    else:
        print(f"[WARN] No se detectó el tablero en {fname}")

cv2.destroyAllWindows()

# calibrar
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("[INFO] Calibración terminada.")
print("camera_matrix:\n", camera_matrix)
print("dist_coeffs:\n", dist_coeffs)

# guardar a disco
np.savez("camera_params.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("[INFO] Guardado camera_params.npz")
