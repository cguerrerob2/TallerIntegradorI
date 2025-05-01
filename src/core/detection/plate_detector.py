import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path="models/plate_detector.pt"):
        self.model = YOLO(model_path)

    def __call__(self, image_bgr, conf=0.5):
        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(rgb_img, conf=conf)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes
