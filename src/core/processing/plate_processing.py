from src.core.detection.plate_detector import PlateDetector
from src.core.processing.superresolution import enhance_plate
from src.core.ocr.recognizer import recognize_plate
import cv2
import numpy as np

_detector = None

def get_plate_detector():
    global _detector
    if _detector is None:
        _detector = PlateDetector(model_path="models/yolov8n.pt")
    return _detector

def process_plate(vehicle_bgr):
    """
    1) Detecta la placa con YOLO,
    2) Aplica superresolución,
    3) Ejecuta OCR para extraer el texto.
    
    Versión mejorada con fallback y detección más robusta.
    """
    try:
        # Si la imagen es muy pequeña, redimensionar
        if vehicle_bgr.shape[0] < 100 or vehicle_bgr.shape[1] < 100:
            scale_factor = max(100 / vehicle_bgr.shape[0], 100 / vehicle_bgr.shape[1])
            vehicle_bgr = cv2.resize(vehicle_bgr, None, fx=scale_factor, fy=scale_factor, 
                                   interpolation=cv2.INTER_LINEAR)
        
        # 1. Intentar detectar la placa con YOLO
        det = get_plate_detector()
        results = det(vehicle_bgr, conf=0.4)
        
        # Si falla la detección, usar todo el vehículo
        if not results or len(results) == 0:
            print("No se detectó placa, usando todo el vehículo")
            # Buscar región más probable de placa (parte frontal/inferior)
            h, w = vehicle_bgr.shape[:2]
            x1, y1 = 0, int(h * 0.7)  # Empezar desde 70% de la altura
            x2, y2 = w, h  # Hasta el final
            plate_region = vehicle_bgr[y1:y2, x1:x2]
            
            # Si la región es válida, usar esta
            if plate_region.size > 0 and plate_region.shape[0] > 0 and plate_region.shape[1] > 0:
                bbox = (x1, y1, x2, y2)
                plate_img = plate_region
            else:
                # Usar todo el vehículo como fallback
                bbox = (0, 0, w, h)
                plate_img = vehicle_bgr
        else:
            # Usar la detección con mayor confianza
            highest_conf = -1
            best_bbox = None
            
            for i, detection in enumerate(results):
                boxes = detection.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = box.conf[0]
                    
                    if conf > highest_conf:
                        highest_conf = conf
                        best_bbox = (x1, y1, x2, y2)
            
            if best_bbox:
                x1, y1, x2, y2 = best_bbox
                # Asegurar que las coordenadas son válidas
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(vehicle_bgr.shape[1], x2), min(vehicle_bgr.shape[0], y2)
                bbox = (x1, y1, x2, y2)
                
                # Extraer la región de la placa
                plate_img = vehicle_bgr[y1:y2, x1:x2]
            else:
                print("Detección sin resultados válidos")
                bbox = (0, 0, vehicle_bgr.shape[1], vehicle_bgr.shape[0])
                plate_img = vehicle_bgr
        
        # 2. Aplicar superresolución
        if plate_img.size > 0 and plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
            plate_sr = enhance_plate(plate_img)
            
            # 3. OCR para extraer el texto
            ocr_text = recognize_plate(plate_sr)
            
            return bbox, plate_sr, ocr_text
        else:
            print("Región de placa inválida")
            return None, vehicle_bgr, "ERROR"
            
    except Exception as e:
        print(f"Error en process_plate: {e}")
        import traceback
        traceback.print_exc()
        return None, vehicle_bgr, "ERROR"