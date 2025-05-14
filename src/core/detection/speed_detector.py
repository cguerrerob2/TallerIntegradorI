import cv2
import numpy as np
import time
from collections import defaultdict
import torch

class SpeedDetector:
    def __init__(self, distance_meters=10, fps=30):
        """
        Inicializa el detector de velocidad para vehículos.
        Ya no carga un modelo propio, usa las detecciones existentes.
        
        Args:
            distance_meters: Distancia estimada en metros que cubre el ancho de la imagen
            fps: Cuadros por segundo del video
        """
        # Configuración
        self.distance_meters = distance_meters
        self.fps = fps
        self.frame_time = 1 / fps
        
        # Diccionario para seguimiento de objetos
        # id_objeto -> [posición_anterior, timestamp_anterior, velocidad_actual]
        self.tracked_objects = defaultdict(lambda: [None, 0, 0])
        
        # Contador para asignar IDs a los objetos detectados
        self.next_id = 0
        
        print("Detector de velocidad inicializado correctamente")
    
    def update(self, detections, frame_shape, pixel_per_meter=None):
        """
        Actualiza velocidades basado en detecciones existentes.
        
        Args:
            detections: Lista de detecciones (x1, y1, x2, y2, cls_id) de otro detector
            frame_shape: Tupla (height, width) del frame
            pixel_per_meter: Relación de píxeles por metro (opcional)
            
        Returns:
            Lista de tuplas (x1, y1, x2, y2, cls_id, object_id, speed_kmh)
        """
        img_h, img_w = frame_shape[:2]
        
        # Si no se proporciona, calcular la relación píxel/metro
        if pixel_per_meter is None:
            pixel_per_meter = img_w / self.distance_meters
        
        # Timestamp actual
        current_time = time.time()
        
        # Lista para las detecciones con velocidad
        speed_detections = []
        
        # Objetos detectados en este frame
        current_objects = []
        
        # Procesar cada detección existente
        for detection in detections:
            x1, y1, x2, y2, cls_id = detection[:5]  # Extraer coordenadas y clase
            
            # Solo procesar vehículos (coche=2, camión=7, autobús=5, moto=3)
            if cls_id in [2, 3, 5, 7]:
                # Centro del objeto para tracking
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_pos = np.array([center_x, center_y])
                
                # Intentar asociar con objetos previos
                best_id = None
                min_dist = float('inf')
                
                # Buscar el objeto más cercano en los ya trackados
                for obj_id, (prev_pos, _, _) in self.tracked_objects.items():
                    if prev_pos is not None:
                        dist = np.linalg.norm(current_pos - prev_pos)
                        if dist < min_dist and dist < img_w * 0.1:  # Máx 10% del ancho
                            min_dist = dist
                            best_id = obj_id
                
                # Si no encontramos coincidencia, crear nuevo ID
                if best_id is None:
                    best_id = self.next_id
                    self.next_id += 1
                    speed_kmh = 0  # Velocidad inicial
                else:
                    # Calcular velocidad si tenemos posición previa
                    prev_pos, prev_time, _ = self.tracked_objects[best_id]
                    
                    # Calcular distancia en metros
                    distance_pixels = np.linalg.norm(current_pos - prev_pos)
                    distance_meters = distance_pixels / pixel_per_meter
                    
                    # Calcular tiempo transcurrido
                    time_delta = current_time - prev_time
                    
                    # Evitar división por cero
                    if time_delta > 0:
                        # Velocidad en metros/segundo
                        speed_mps = distance_meters / time_delta
                        # Convertir a km/h
                        speed_kmh = speed_mps * 3.6
                        
                        # Suavizar la velocidad (media móvil)
                        prev_speed = self.tracked_objects[best_id][2]
                        if prev_speed > 0:
                            speed_kmh = 0.7 * speed_kmh + 0.3 * prev_speed
                    else:
                        speed_kmh = 0
                
                # Actualizar tracking
                self.tracked_objects[best_id] = [current_pos, current_time, speed_kmh]
                
                # Añadir a la lista de objetos actuales
                current_objects.append(best_id)
                
                # Añadir a resultados
                speed_detections.append((x1, y1, x2, y2, cls_id, best_id, speed_kmh))
        
        # Limpiar objetos que ya no se ven (después de cierto tiempo)
        current_time_threshold = current_time - 2.0  # 2 segundos
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in current_objects and self.tracked_objects[obj_id][1] < current_time_threshold:
                del self.tracked_objects[obj_id]
        
        return speed_detections

    def draw_results(self, frame, speed_detections):
        """
        Dibuja los resultados de velocidad en el frame.
        
        Args:
            frame: Imagen BGR
            speed_detections: Lista de (x1, y1, x2, y2, cls_id, object_id, speed_kmh)
            
        Returns:
            Frame con anotaciones
        """
        annotated_frame = frame.copy()
        
        for (x1, y1, x2, y2, cls_id, obj_id, speed_kmh) in speed_detections:
            # Determinar color según velocidad
            if speed_kmh < 30:
                color = (0, 255, 0)  # Verde para velocidad baja
            elif speed_kmh < 60:
                color = (0, 255, 255)  # Amarillo para velocidad media
            else:
                color = (0, 0, 255)  # Rojo para velocidad alta
            
            # Dibujar rectángulo
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Etiquetar según la clase
            if cls_id == 2:
                label = "Car"
            elif cls_id == 3:
                label = "Motorcycle"
            elif cls_id == 5:
                label = "Bus"
            elif cls_id == 7:
                label = "Truck"
            else:
                label = f"Class {cls_id}"
            
            # Añadir velocidad a la etiqueta
            label = f"{label}: {speed_kmh:.1f} km/h"
            
            # Dibujar texto con fondo
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 10, y1),
                         color, -1)
            cv2.putText(annotated_frame, label, 
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame

    def calibrate(self, detections, reference_object_length=4.5):
        """
        Calibra la relación píxel/metro usando detecciones existentes.
        
        Args:
            detections: Lista de detecciones (x1, y1, x2, y2, cls_id)
            reference_object_length: Longitud promedio de un auto (4.5m)
            
        Returns:
            píxels_por_metro
        """
        # Buscar el objeto más grande
        max_area = 0
        reference_width = 0
        
        for detection in detections:
            x1, y1, x2, y2, cls_id = detection[:5]
            
            # Solo considerar vehículos
            if cls_id in [2, 5, 7]:  # car, bus, truck
                width = x2 - x1
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    reference_width = width
        
        if reference_width > 0:
            # Calcular píxeles por metro
            pixels_per_meter = reference_width / reference_object_length
            return pixels_per_meter
        else:
            # Valor por defecto
            return None