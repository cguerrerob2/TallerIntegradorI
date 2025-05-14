from src.core.detection.plate_detector import PlateDetector
from src.core.processing.superresolution import enhance_plate
from src.core.ocr.recognizer import recognize_plate

_detector = None

def get_plate_detector():
    global _detector
    if _detector is None:
        _detector = PlateDetector("models/yolov8n.pt")
    return _detector

def process_plate(vehicle_bgr):
    """
    1) Detecta la placa con YOLO,
    2) Aplica superresolución (EDSR),
    3) Ejecuta OCR para extraer el texto.
    """
    det = get_plate_detector()
    bboxes = det(vehicle_bgr, conf=0.5)
    if not bboxes:
        return None, None, None
    x1, y1, x2, y2 = bboxes[0]
    plate_crop = vehicle_bgr[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None, None, None
    plate_sr = enhance_plate(plate_crop)
    ocr_text = recognize_plate(plate_sr)
    return bboxes[0], plate_sr, ocr_text
