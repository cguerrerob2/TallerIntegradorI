import cv2
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_bgr, draw=False):
        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(rgb_img, conf=0.3)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                detections.append((x1, y1, x2, y2, cls_id))
        if draw:
            for (x1, y1, x2, y2, cls_id) in detections:
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
        return detections
