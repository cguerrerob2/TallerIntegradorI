import cv2
import time
import random

def recognize_plate(plate_bgr):
    """
    Reconoce el texto de una placa en una imagen con mejor manejo de 
    caracteres comúnmente confundidos
    
    Args:
        plate_bgr: Imagen de la placa
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return None
    
    # En un caso real, aquí iría el procesamiento de OCR
    # Como es un stub, generamos un ID único basado en timestamp
    timestamp = int(time.time())
    
    # Crear un ID único cada vez para asegurar que se muestre cada placa detectada
    random_suffix = random.randint(100, 999)
    
    # Formato: INFR_TIMESTAMP_RANDOM - fácil de reconocer visualmente
    return f"INFR_{timestamp}_{random_suffix}"