import cv2
import torch
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Cargar modelo con configuración optimizada
        self.model = YOLO(model_path)
        
        # Dispositivo óptimo (GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tamaño óptimo de procesamiento
        self.imgsz = 320  # Tamaño más pequeño para detección más rápida
        
        # Caché para evitar procesar frames similares
        self.last_frame_hash = None
        self.last_detections = []

    def detect(self, image_bgr, draw=False):
        # 1. Verificar si el frame es muy similar al anterior usando hash perceptual
        if image_bgr.shape[0] > 200:  # Solo para imágenes grandes
            small = cv2.resize(image_bgr, (8, 8))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            img_hash = hash(gray.tobytes())
            
            if hasattr(self, 'last_frame_hash') and self.last_frame_hash == img_hash:
                return self.last_detections
            
            self.last_frame_hash = img_hash
        
        # 2. Procesar en tamaño reducido para mayor velocidad
        orig_shape = image_bgr.shape
        if orig_shape[0] > self.imgsz or orig_shape[1] > self.imgsz:
            scale = min(self.imgsz / orig_shape[0], self.imgsz / orig_shape[1])
            new_shape = (int(orig_shape[1] * scale), int(orig_shape[0] * scale))
            resized = cv2.resize(image_bgr, new_shape)
            results = self.model.predict(resized, conf=0.35, verbose=False)
            scale_factor = (orig_shape[1] / new_shape[0], orig_shape[0] / new_shape[1])
        else:
            results = self.model.predict(image_bgr, conf=0.35, verbose=False)
            scale_factor = (1.0, 1.0)
        
        # 3. Extraer detecciones
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Escalar coordenadas al tamaño original
                if scale_factor != (1.0, 1.0):
                    x1, x2 = int(x1 * scale_factor[0]), int(x2 * scale_factor[0])
                    y1, y2 = int(y1 * scale_factor[1]), int(y2 * scale_factor[1])
                
                cls_id = int(box.cls[0])
                detections.append((x1, y1, x2, y2, cls_id))
        
        # 4. Guardar en caché
        self.last_detections = detections
        
        # 5. Dibujar si es necesario
        if draw:
            for (x1, y1, x2, y2, cls_id) in detections:
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return detections