# config.py

# resolución de previsualización de la cámara de la mano
PREVIEW_W = 640
PREVIEW_H = 480

# reconocimiento en vivo
RECOGNIZE_MODE = True

# umbral de confianza para decir "????"
CONFIDENCE_THRESHOLD = 1.2  # puedes bajarlo si quieres que sea más estricto

# dónde guardamos los gestos (.npz)
GESTURES_DIR = "gestures"

# corrección de distorsión (ojo de pez)
USE_UNDISTORT_HAND = False      # pon True si también calibras la de la mano
USE_UNDISTORT_BOARD = True      # esta es la de la cámara USB (tablero)

# rutas de los parámetros de cámara
HAND_CAMERA_PARAMS_PATH = "camera_params_hand.npz"
BOARD_CAMERA_PARAMS_PATH = "camera_params_board.npz"
