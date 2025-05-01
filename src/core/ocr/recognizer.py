import cv2

def recognize_plate(plate_bgr):
    """
    Simulación de OCR:
    - Si la imagen no está vacía, devuelve "ABC123".
    - Si está vacía, retorna None.
    """
    if plate_bgr.size == 0:
        return None
    return "ABC123"
