import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        print("Modelo de detección de placas inicializado correctamente")

    def __call__(self, image_bgr, conf=0.4):
        # Procesar solo una región específica donde es probable encontrar placas
        # (parte inferior de los vehículos)
        h, w = image_bgr.shape[:2]
        roi = image_bgr[h//2:, :]  # Solo la mitad inferior de la imagen
        
        results = self.model.predict(roi, conf=conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Ajustar coordenadas Y a la imagen original
                y1 += h//2
                y2 += h//2
                boxes.append((x1, y1, x2, y2))
        
        return boxes