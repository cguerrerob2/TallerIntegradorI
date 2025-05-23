import cv2
import numpy as np
import time
import os
from collections import defaultdict, deque
from datetime import datetime

class AccidentDetector:
    def __init__(self, time_window=1.0, cooldown=3.0, save_dir="data/output"):
        """
        Detector de accidentes de trafico mejorado (choques y atropellos).
        
        Args:
            time_window: Ventana de tiempo (segundos) para analizar cambios bruscos de velocidad
            cooldown: Tiempo (segundos) minimo entre alertas de accidente
            save_dir: Directorio para guardar las capturas de accidentes
        """
        # Configuración
        self.time_window = time_window
        self.cooldown_time = cooldown
        
        # Directorio para guardar capturas de accidentes
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Seguimiento de objetos y sus trayectorias
        self.tracked_vehicles = {}
        self.tracked_pedestrians = {}
        
        # Historial de posiciones y velocidades para cada vehículo y peatón
        self.position_history = defaultdict(lambda: deque(maxlen=30))
        self.velocity_history = defaultdict(lambda: deque(maxlen=30))
        
        # Lista de accidentes detectados recientemente
        self.recent_accidents = []
        self.last_accident_time = 0
        
        # Registro de accidentes para no duplicar
        self.registered_accidents = set()
        self.saved_accidents = set()
        
        # Contador de frames
        self.frame_count = 0
        
        # Historial de aceleraciones para detección más robusta
        self.acceleration_history = defaultdict(lambda: deque(maxlen=15))
        
        # Variables para configurar sensibilidad
        self.collision_distance_factor = 0.12  # Menor valor = mayor sensibilidad
        self.deceleration_threshold = 150      # Valor más alto = menor sensibilidad
        self.pedestrian_proximity_threshold = 0.1  # Umbral para atropellos (proporción del tamaño)
        
        # Tiempo mínimo para considerar un seguimiento estable
        self.min_tracking_time = 0.5  # segundos
        
        # Variables para filtrado temporal
        self.accident_confidence = {}
        self.confidence_threshold = 6  # Aumentado para reducir falsos positivos
        
        # Umbrales de área para filtrar falsas detecciones
        self.min_vehicle_area = 2500   # Área mínima para considerar un vehículo (px²)
        self.max_overlap_ratio = 0.8   # Máxima superposición permitida con otro objeto
        
        print(f"Detector de accidentes mejorado inicializado - Guardando en: {self.save_dir}")
    
    def update(self, detections, speed_detections, frame_shape):
        """
        Actualiza el detector con nuevas detecciones y analiza posibles accidentes.
        
        Args:
            detections: Lista de detecciones (x1, y1, x2, y2, cls_id)
            speed_detections: Lista de (x1, y1, x2, y2, cls_id, obj_id, speed_kmh)
            frame_shape: Dimensiones del frame (h, w)
            
        Returns:
            Lista de accidentes: incluye choques y atropellos
        """
        current_time = time.time()
        h, w = frame_shape[:2]
        self.frame_count += 1
        
        # Mapa de objetos detectados en este frame
        current_vehicles = {}
        current_pedestrians = {}
        
        # Procesar todas las detecciones primero para identificar peatones
        for detection in detections:
            try:
                x1, y1, x2, y2, cls_id = detection[:5]
                
                # Convertir coordenadas a enteros
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ignorar detecciones muy pequeñas (posibles falsos positivos)
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Filtrar por tamaño mínimo (más estricto para peatones)
                if width < 25 or height < 25 or area < 800:
                    continue
                
                # Centro del objeto
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Solo procesar peatones (clase 0 en YOLO)
                if cls_id == 0:
                    # Verificar que el peatón no esté en los bordes de la imagen (poca confianza)
                    if x1 < 10 or y1 < 10 or x2 > w-10 or y2 > h-10:
                        continue
                        
                    # Usar una combinación de posición y tamaño como ID improvisado
                    obj_id = f"ped_{center_x}_{center_y}_{self.frame_count % 100}"
                    
                    # Buscar si hay un peatón cercano ya en tracking
                    closest_id = None
                    min_distance = float('inf')
                    for pid, ped in self.tracked_pedestrians.items():
                        if time.time() - ped['time'] < 1.0:  # Solo considerar los recientes
                            dist = np.linalg.norm(np.array([center_x, center_y]) - ped['position'])
                            if dist < min_distance and dist < max(width, height) * 1.2:
                                min_distance = dist
                                closest_id = pid
                    
                    # Si encontramos un peatón cercano, usar su ID
                    if closest_id is not None:
                        obj_id = closest_id
                    
                    # Almacenar información del peatón
                    current_pedestrians[obj_id] = {
                        'id': obj_id,
                        'position': np.array([center_x, center_y]),
                        'size': np.array([width, height]),
                        'bbox': (x1, y1, x2, y2),
                        'class': cls_id,
                        'speed': 0,  # Inicialmente 0
                        'time': current_time,
                        'has_collided': False,
                        'last_positions': deque(maxlen=10),
                        'confidence': 0.7 if obj_id in self.tracked_pedestrians else 0.4,  # Confianza inicial más baja
                        'area': area,
                        'frames_detected': 1 if obj_id not in self.tracked_pedestrians else self.tracked_pedestrians[obj_id].get('frames_detected', 0) + 1
                    }
                    
                    # Actualizar historial de posiciones
                    if obj_id in self.tracked_pedestrians:
                        prev_pos = self.tracked_pedestrians[obj_id]['position']
                        prev_time = self.tracked_pedestrians[obj_id]['time']
                        time_delta = max(0.001, current_time - prev_time)
                        
                        # Calcular velocidad aproximada del peatón
                        if time_delta > 0:
                            velocity = (current_pedestrians[obj_id]['position'] - prev_pos) / time_delta
                            speed_value = np.linalg.norm(velocity)
                            
                            # Filtrar velocidades irrealistas para peatones
                            if speed_value < 150:  # Umbral en píxeles por segundo
                                current_pedestrians[obj_id]['speed'] = speed_value
                            
                        # Transferir el historial de posiciones y estado de colisión
                        if 'last_positions' in self.tracked_pedestrians[obj_id]:
                            current_pedestrians[obj_id]['last_positions'] = self.tracked_pedestrians[obj_id]['last_positions'].copy()
                        current_pedestrians[obj_id]['last_positions'].append(current_pedestrians[obj_id]['position'])
                        
                        if 'has_collided' in self.tracked_pedestrians[obj_id]:
                            current_pedestrians[obj_id]['has_collided'] = self.tracked_pedestrians[obj_id]['has_collided']
                            
                        # Aumentar confianza por trackeo continuo (con límite superior)
                        if 'confidence' in self.tracked_pedestrians[obj_id]:
                            current_pedestrians[obj_id]['confidence'] = min(
                                0.95, self.tracked_pedestrians[obj_id]['confidence'] + 0.05
                            )
                    else:
                        # Nuevo peatón - inicializar historial
                        current_pedestrians[obj_id]['last_positions'].append(current_pedestrians[obj_id]['position'])
                        
            except Exception as e:
                print(f"Error al procesar deteccion de peaton: {e}")
        
        # Procesar detecciones con velocidad (vehículos)
        for detection in speed_detections:
            try:
                x1, y1, x2, y2, cls_id, obj_id, speed_kmh = detection
                
                # Convertir coordenadas a enteros
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ignorar detecciones muy pequeñas
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Filtrar objetos pequeños o demasiado grandes (posibles errores)
                if width < 30 or height < 30 or area < self.min_vehicle_area or area > (h * w * 0.5):
                    continue
                
                # Centro del objeto
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Verificar que esté dentro de los límites del frame (con margen)
                if not (5 <= center_x < w-5 and 5 <= center_y < h-5):
                    continue
                
                # Almacenar información actual
                current_vehicles[obj_id] = {
                    'id': obj_id,
                    'position': np.array([center_x, center_y]),
                    'size': np.array([width, height]),
                    'bbox': (x1, y1, x2, y2),
                    'class': cls_id,
                    'speed': speed_kmh,
                    'time': current_time,
                    'has_collided': False,
                    'deformation': 0,
                    'confidence': 0.9 if obj_id in self.tracked_vehicles else 0.6,
                    'area': area,
                    'frames_detected': 1 if obj_id not in self.tracked_vehicles else self.tracked_vehicles[obj_id].get('frames_detected', 0) + 1
                }
                
                # Actualizar historial de posiciones
                if obj_id in self.tracked_vehicles:
                    prev_pos = self.tracked_vehicles[obj_id]['position']
                    prev_time = self.tracked_vehicles[obj_id]['time']
                    time_delta = max(0.001, current_time - prev_time)
                    
                    # Calcular vector de velocidad (píxeles/frame)
                    velocity = (current_vehicles[obj_id]['position'] - prev_pos) / time_delta
                    
                    # Calcular aceleración si tenemos suficiente historial
                    if len(self.velocity_history[obj_id]) > 0:
                        prev_velocity = self.velocity_history[obj_id][-1]
                        acceleration = (velocity - prev_velocity) / time_delta
                        self.acceleration_history[obj_id].append(acceleration)
                    
                    # Almacenar historial
                    self.position_history[obj_id].append(current_vehicles[obj_id]['position'])
                    self.velocity_history[obj_id].append(velocity)
                    
                    # Transferir estado de colisión anterior y otros atributos persistentes
                    if 'has_collided' in self.tracked_vehicles[obj_id]:
                        current_vehicles[obj_id]['has_collided'] = self.tracked_vehicles[obj_id]['has_collided']
                    if 'deformation' in self.tracked_vehicles[obj_id]:
                        current_vehicles[obj_id]['deformation'] = self.tracked_vehicles[obj_id]['deformation']
                    
                    # Aumentar confianza por trackeo continuo
                    if 'confidence' in self.tracked_vehicles[obj_id]:
                        current_vehicles[obj_id]['confidence'] = min(
                            0.99, self.tracked_vehicles[obj_id]['confidence'] + 0.05
                        )
                else:
                    # Nuevo objeto, inicializar historial
                    self.position_history[obj_id].append(current_vehicles[obj_id]['position'])
                    self.velocity_history[obj_id].append(np.array([0, 0]))
                    self.acceleration_history[obj_id].append(np.array([0, 0]))
                    
            except Exception as e:
                print(f"Error al procesar deteccion de vehiculo: {e}")
        
        # Actualizar objetos trackados
        self.tracked_vehicles = current_vehicles
        self.tracked_pedestrians = current_pedestrians
        
        # Lista de accidentes detectados
        accidents = []
        
        # 1. DETECCIÓN DE COLISIONES ENTRE VEHÍCULOS
        try:
            vehicle_ids = list(current_vehicles.keys())
            for i in range(len(vehicle_ids)):
                for j in range(i + 1, len(vehicle_ids)):
                    id1, id2 = vehicle_ids[i], vehicle_ids[j]
                    v1, v2 = current_vehicles[id1], current_vehicles[id2]
                    
                    # Verificar si alguno de los vehículos tiene baja confianza o pocos frames detectados
                    # Aumentamos los umbrales para mayor seguridad
                    if (v1.get('confidence', 0) < 0.9 or 
                        v2.get('confidence', 0) < 0.9 or
                        v1.get('frames_detected', 0) < 10 or
                        v2.get('frames_detected', 0) < 10):
                        continue
                        
                    # Verificar intersección real de bounding boxes (REQUISITO OBLIGATORIO)
                    bb1 = v1['bbox']
                    bb2 = v2['bbox']
                    
                    # Comprobar intersección de bounding boxes
                    intersect = not (bb1[2] < bb2[0] or bb1[0] > bb2[2] or bb1[3] < bb2[1] or bb1[1] > bb2[3])
                    
                    # Calcular área de intersección
                    if not intersect:
                        continue  # Si no hay intersección real, no hay colisión
                        
                    # Calcular porcentaje de intersección
                    x_left = max(bb1[0], bb2[0])
                    y_top = max(bb1[1], bb2[1])
                    x_right = min(bb1[2], bb2[2])
                    y_bottom = min(bb1[3], bb2[3])
                    
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
                    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
                    
                    # Exigir una superposición significativa - al menos 20% de área de uno de los objetos
                    if intersection_area < min(bb1_area, bb2_area) * 0.2:
                        continue
                    
                    # Verificar si al menos uno de los vehículos está en movimiento significativo
                    v1_moving = v1['speed'] > 15 or (len(self.velocity_history[id1]) > 3 and 
                                                np.linalg.norm(self.velocity_history[id1][-1]) > 10)
                    v2_moving = v2['speed'] > 15 or (len(self.velocity_history[id2]) > 3 and 
                                                np.linalg.norm(self.velocity_history[id2][-1]) > 10)
                    
                    # SOLO detectar colisión si hay intersección real y al menos uno está en movimiento
                    if intersect and (v1_moving or v2_moving):
                        collision_id = f"{min(id1, id2)}_{max(id1, id2)}"
                        
                        # Verificar si alguno de los objetos ha estado estacionario durante mucho tiempo
                        v1_stationary = self._is_object_stationary(id1, 10)
                        v2_stationary = self._is_object_stationary(id2, 10)
                        
                        # Si ambos son estacionarios, es muy probable un falso positivo
                        if v1_stationary and v2_stationary:
                            continue
                        
                        # Si ya está registrado, incrementar contador de confianza
                        if collision_id in self.accident_confidence:
                            self.accident_confidence[collision_id] += 1
                        else:
                            self.accident_confidence[collision_id] = 1
                        
                        # Solo registrar si supera el umbral de confianza
                        if self.accident_confidence[collision_id] >= self.confidence_threshold:
                            # Evitar duplicados de la misma colisión en un periodo corto
                            if collision_id not in self.registered_accidents:
                                # Marcar ambos vehículos como colisionados
                                current_vehicles[id1]['has_collided'] = True
                                current_vehicles[id2]['has_collided'] = True
                                
                                # Crear un bbox que envuelva ambos vehículos
                                x_min = min(v1['bbox'][0], v2['bbox'][0])
                                y_min = min(v1['bbox'][1], v2['bbox'][1])
                                x_max = max(v1['bbox'][2], v2['bbox'][2])
                                y_max = max(v1['bbox'][3], v2['bbox'][3])
                                
                                # Registrar el accidente como colisión confirmada
                                accident_data = {
                                    'type': 'collision',
                                    'vehicles': [id1, id2],
                                    'risk': 100,
                                    'bbox': (x_min, y_min, x_max, y_max),
                                    'time': current_time,
                                    'collision_id': collision_id
                                }
                                
                                accidents.append(accident_data)
                                self.registered_accidents.add(collision_id)
                                print(f"¡COLISION DETECTADA! entre vehiculos {id1} y {id2}")
                    else:
                        # Si no hay colisión, reducir contador de confianza
                        collision_id = f"{min(id1, id2)}_{max(id1, id2)}"
                        if collision_id in self.accident_confidence:
                            self.accident_confidence[collision_id] = max(0, self.accident_confidence[collision_id] - 0.5)
        except Exception as e:
            print(f"Error en deteccion de colisiones: {e}")
        
        # 2. DETECTAR ATROPELLOS (COLISIONES VEHÍCULO-PEATÓN)
        try:
            for vehicle_id, vehicle in current_vehicles.items():
                for pedestrian_id, pedestrian in current_pedestrians.items():
                    # Verificar si ya han estado en una colisión
                    if vehicle.get('has_collided', False) and pedestrian.get('has_collided', False):
                        continue
                    
                    # Solo procesar si ambos tienen buena confianza y suficiente historial
                    if (vehicle.get('confidence', 0) < 0.85 or 
                        pedestrian.get('confidence', 0) < 0.8 or
                        vehicle.get('frames_detected', 0) < 5 or
                        pedestrian.get('frames_detected', 0) < 5):
                        continue
                        
                    # Verificar intersección de bounding boxes
                    vbb = vehicle['bbox']
                    pbb = pedestrian['bbox']
                    
                    # Calcular IoU (Intersection over Union)
                    x_left = max(vbb[0], pbb[0])
                    y_top = max(vbb[1], pbb[1])
                    x_right = min(vbb[2], pbb[2])
                    y_bottom = min(vbb[3], pbb[3])
                    
                    # Si no hay intersección real, no es un atropello
                    if x_right < x_left or y_bottom < y_top:
                        continue
                    
                    # Calcular área de intersección
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    pedestrian_area = pedestrian['area']
                    
                    # Exigir una superposición significativa (al menos 30% del peatón debe estar superpuesto)
                    if intersection_area < pedestrian_area * 0.3:
                        continue
                    
                    # Verificar que el vehículo está en movimiento (velocidad significativa)
                    vehicle_moving = vehicle['speed'] > 15 or (
                        len(self.velocity_history[vehicle_id]) > 3 and 
                        np.linalg.norm(self.velocity_history[vehicle_id][-1]) > 10
                    )
                    
                    # Si el vehículo no se mueve, no es un atropello
                    if not vehicle_moving:
                        continue
                    
                    # Crear ID único para esta colisión
                    collision_id = f"ped_{pedestrian_id}_{vehicle_id}"
                    
                    # Usar sistema de confianza temporal
                    if collision_id in self.accident_confidence:
                        self.accident_confidence[collision_id] += 1
                    else:
                        self.accident_confidence[collision_id] = 1
                        
                    # Solo registrar si la confianza supera el umbral
                    if self.accident_confidence[collision_id] >= self.confidence_threshold:
                        if collision_id not in self.registered_accidents:
                            # Marcar ambos como colisionados
                            vehicle['has_collided'] = True
                            pedestrian['has_collided'] = True
                            
                            # Crear un bbox que envuelva a ambos con un margen
                            margin = 15
                            x_min = max(0, min(vbb[0], pbb[0]) - margin)
                            y_min = max(0, min(vbb[1], pbb[1]) - margin)
                            x_max = min(w, max(vbb[2], pbb[2]) + margin)
                            y_max = min(h, max(vbb[3], pbb[3]) + margin)
                            
                            # Registrar el atropello
                            accident_data = {
                                'type': 'pedestrian_collision',
                                'vehicles': [vehicle_id],
                                'pedestrians': [pedestrian_id],
                                'risk': 100,  # Atropello confirmado
                                'bbox': (x_min, y_min, x_max, y_max),
                                'time': current_time,
                                'collision_id': collision_id
                            }
                            
                            accidents.append(accident_data)
                            self.registered_accidents.add(collision_id)
                            print(f"¡ATROPELLO DETECTADO! Vehiculo {vehicle_id} y peaton {pedestrian_id}")
                    else:
                        # Si no hay colisión, reducir contador de confianza
                        if collision_id in self.accident_confidence:
                            self.accident_confidence[collision_id] = max(0, self.accident_confidence[collision_id] - 0.5)
        except Exception as e:
            print(f"Error en deteccion de atropellos: {e}")
        
        # 4. Guardar accidentes recientes para alertas y registro
        for accident in accidents:
            self.recent_accidents.append(accident)
                
            # Limitar tamaño de la lista de accidentes recientes
            if len(self.recent_accidents) > 10:
                self.recent_accidents.pop(0)
        
        return accidents
    
    def _is_object_stationary(self, obj_id, num_frames=10):
        """
        Verifica si un objeto está estacionario durante los últimos frames
        """
        if obj_id not in self.position_history or len(self.position_history[obj_id]) < num_frames:
            return False
            
        positions = list(self.position_history[obj_id])[-num_frames:]
        if len(positions) < 2:
            return True
            
        total_movement = 0
        for i in range(1, len(positions)):
            movement = np.linalg.norm(positions[i] - positions[i-1])
            total_movement += movement
            
        avg_movement = total_movement / (len(positions) - 1)
        return avg_movement < 2.0  # 2 píxeles promedio por frame
        
    def _calculate_movement(self, positions):
        """
        Calcula el movimiento promedio entre posiciones
        """
        if len(positions) < 2:
            return 0
            
        total_movement = 0
        for i in range(1, len(positions)):
            movement = np.linalg.norm(positions[i] - positions[i-1])
            total_movement += movement
            
        return total_movement / (len(positions) - 1)
    
    def draw_results(self, frame, accidents):
        """
        Dibuja los resultados del analisis de accidentes en el frame.
        
        Args:
            frame: Imagen BGR
            accidents: Lista de detecciones de accidentes
            
        Returns:
            Frame con anotaciones
        """
        if frame is None:
            return None
            
        annotated_frame = frame.copy()
        
        # Visualizar tracker de vehículos y peatones con alta confianza (opcional para debug)
        for veh_id, veh in self.tracked_vehicles.items():
            if veh.get('confidence', 0) > 0.8:  # Solo mostrar los de alta confianza
                x1, y1, x2, y2 = veh['bbox']
                # Dibujamos solo el rectángulo sin texto de velocidad
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # Añadir etiqueta solo con el tipo de vehículo
                label = "CAR"
                cv2.putText(annotated_frame, label, (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        for ped_id, ped in self.tracked_pedestrians.items():
            if ped.get('confidence', 0) > 0.8:  # Solo mostrar los de alta confianza
                x1, y1, x2, y2 = ped['bbox']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # Etiqueta para peatones
                label = "PERSON"
                cv2.putText(annotated_frame, label, (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Mejora visual: Añadir marcador destacado para accidentes
        for accident in accidents:
            try:
                x1, y1, x2, y2 = accident['bbox']
                accident_type = accident['type']
                
                # Convertir coordenadas a enteros
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Guardar captura del accidente
                if 'collision_id' in accident:
                    self.save_accident_image(frame, accident)
                        
                # Color según tipo de accidente
                if accident_type == 'pedestrian_collision':
                    color = (0, 0, 255)  # Rojo para atropellos
                    label = "ATROPELLO DETECTADO!"
                else:
                    color = (0, 0, 255)  # Rojo para otros accidentes
                    label = "ACCIDENTE DETECTADO!"
                
                # Dibujar un círculo para llamar la atención
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = max((x2 - x1), (y2 - y1)) // 3
                cv2.circle(annotated_frame, (center_x, center_y), radius, color, 2)
                
                # Dibujar rectángulo alrededor del área del accidente con línea más gruesa
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Dibujar texto con fondo más destacado
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - text_size[1] - 15), 
                            (x1 + text_size[0] + 15, y1),
                            color, -1)
                cv2.putText(annotated_frame, label, 
                        (x1 + 8, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error al dibujar accidente: {e}")
        
        # Mostrar accidentes recientes durante un tiempo con una alerta más grande y visible
        try:
            current_time = time.time()
            for accident in self.recent_accidents:
                # Mostrar los accidentes recientes por 5 segundos
                if current_time - accident['time'] < 5.0:
                    # Color según tipo de accidente
                    alert_color = (0, 0, 200)
                    border_color = (0, 0, 255)
                    
                    if accident['type'] == 'pedestrian_collision':
                        alert_text = "ALERTA! ATROPELLO A PEATON DETECTADO"
                    else:
                        alert_text = "ALERTA! ACCIDENTE DE TRAFICO DETECTADO"
                    
                    # Dibujar alerta general en la parte inferior para no superponerse con otros textos
                    h, w = annotated_frame.shape[:2]
                    
                    # Calcular posición vertical para evitar superposición
                    alert_y = h - 120  # Colocar en la parte inferior
                    
                    cv2.rectangle(annotated_frame, 
                                (10, alert_y), 
                                (w - 10, alert_y + 90),
                                alert_color, -1)
                    
                    cv2.rectangle(annotated_frame, 
                                (15, alert_y + 5), 
                                (w - 15, alert_y + 85),
                                border_color, 4)
                    
                    # Texto más grande y destacado
                    cv2.putText(annotated_frame, alert_text, 
                            (30, alert_y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        except Exception as e:
            print(f"Error al mostrar alertas recientes: {e}")
        
        return annotated_frame

    def save_accident_image(self, frame, accident):
        """
        Guarda una imagen del accidente detectado.
        
        Args:
            frame: Imagen completa del frame actual
            accident: Datos del accidente
        """
        # Solo guardar accidentes confirmados
        if accident['risk'] < 100:
            return
                
        # Evitar guardar múltiples veces el mismo accidente
        if accident['collision_id'] in self.saved_accidents:
            return
        
        try:
            # Verificar que frame no sea None
            if frame is None:
                print("ERROR: Frame es None, no se puede guardar imagen")
                return
                
            # Marcar como guardado
            self.saved_accidents.add(accident['collision_id'])
            
            # Crear nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Añadir información del tipo de accidente
            accident_type = "atropello" if accident['type'] == "pedestrian_collision" else "accidente"
            
            # Crear nombre de archivo simple
            filename = f"{self.save_dir}/{accident_type}_{timestamp}.jpg"
            
            # Obtener el área de la colisión
            x1, y1, x2, y2 = accident['bbox']
            
            # Convertir a enteros
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calcular centro para verificar si es un buen recorte
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Si el centro está fuera del frame o muy cerca de los bordes, ajustar
            h, w = frame.shape[:2]
            if center_x < 10 or center_x > w-10 or center_y < 10 or center_y > h-10:
                # Centrar mejor el recorte
                width = x2 - x1
                height = y2 - y1
                center_x = max(width//2 + 20, min(w - width//2 - 20, center_x))
                center_y = max(height//2 + 20, min(h - height//2 - 20, center_y))
                x1 = max(0, center_x - width//2 - 20)
                y1 = max(0, center_y - height//2 - 20)
                x2 = min(w, center_x + width//2 + 20)
                y2 = min(h, center_y + height//2 + 20)
            
            # Expandir el área para incluir más contexto balanceado alrededor del accidente
            margin_x = min(100, max(50, int((x2 - x1) * 0.3)))  # Margen proporcional al tamaño
            margin_y = min(100, max(50, int((y2 - y1) * 0.3)))  # Pero limitado entre 50-100px
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            
            # Crear una copia del frame para marcar el área
            accident_image = frame.copy()
            
            # Comprobar si hay vehículos y peatones en la zona de accidente para destacarlos
            if accident['type'] == 'pedestrian_collision' and 'vehicles' in accident and 'pedestrians' in accident:
                # Destacar el vehículo y peatón involucrados
                for v_id in accident['vehicles']:
                    if v_id in self.tracked_vehicles:
                        veh = self.tracked_vehicles[v_id]
                        vx1, vy1, vx2, vy2 = veh['bbox']
                        # Resaltar el vehículo en azul
                        cv2.rectangle(accident_image, (vx1, vy1), (vx2, vy2), (255, 0, 0), 3)
                        cv2.putText(accident_image, "VEHICULO", (vx1, vy1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                for p_id in accident['pedestrians']:
                    if p_id in self.tracked_pedestrians:
                        ped = self.tracked_pedestrians[p_id]
                        px1, py1, px2, py2 = ped['bbox']
                        # Resaltar el peatón en verde
                        cv2.rectangle(accident_image, (px1, py1), (px2, py2), (0, 255, 0), 3)
                        cv2.putText(accident_image, "PEATON", (px1, py1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif 'vehicles' in accident:
                # Para accidentes entre vehículos o de un solo vehículo
                for v_id in accident['vehicles']:
                    if v_id in self.tracked_vehicles and v_id != -1:  # -1 es el ID especial para detecciones manuales
                        veh = self.tracked_vehicles[v_id]
                        vx1, vy1, vx2, vy2 = veh['bbox']
                        # Resaltar el vehículo en azul
                        cv2.rectangle(accident_image, (vx1, vy1), (vx2, vy2), (255, 0, 0), 3)
                        cv2.putText(accident_image, "VEHICULO", (vx1, vy1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Marcar la zona del accidente
            cv2.rectangle(accident_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Indicar el tipo correcto de accidente
            if accident['type'] == "pedestrian_collision":
                label = "ATROPELLO DETECTADO"
            else:
                label = "ACCIDENTE DETECTADO"
                
            cv2.putText(accident_image, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Añadir timestamp a la imagen
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(accident_image, time_str, (10, h-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            # Guardar la imagen con las anotaciones
            cv2.imwrite(filename, accident_image)
            print(f"Accidente guardado: {filename}")
            
        except Exception as e:
            print(f"Error al guardar imagen del accidente: {e}")

    def force_accident_detection(self, frame, bbox):
        """
        Forzar la deteccion de un accidente en una ubicacion especifica (para pruebas)
        
        Args:
            frame: Frame actual
            bbox: (x1, y1, x2, y2) del area del accidente
        """
        current_time = time.time()
        accident_id = f"manual_{int(current_time)}"
        
        if accident_id not in self.registered_accidents:
            accident_data = {
                'type': 'collision',
                'vehicles': [-1],  # ID especial para deteccion manual
                'risk': 100,       # Accidente confirmado
                'bbox': bbox,
                'time': current_time,
                'collision_id': accident_id
            }
            
            self.recent_accidents.append(accident_data)
            self.registered_accidents.add(accident_id)
            
            # Guardar captura
            self.save_accident_image(frame, accident_data)
            
            return accident_data
        
        return None