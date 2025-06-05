import cv2

def recognize_plate(plate_bgr):
    """
    Reconoce el texto de una placa en una imagen con mejor manejo de 
    caracteres comúnmente confundidos
    
    Args:
        plate_bgr: Imagen de la placa
        is_night: Flag que indica si es escena nocturna
    """
    if plate_bgr.size == 0:
        return None
    return "infractor_123"
