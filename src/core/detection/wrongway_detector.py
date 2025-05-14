import cv2
import numpy as np
import time
from collections import defaultdict, deque

class WrongWayDetector:
    def __init__(self, allowed_direction='right', detection_threshold=5, detection_zone=None):
        """
        Detector de vehículos en sentido contrario.
        
        Args:
            allowed_direction: Dirección permitida ('right', 'left', 'up', 'down')
            detection_threshold: Cuántos frames de movimiento en sentido contrario para confirmar
            detection_zone: Región de la imagen donde detectar (x1,y1,x2,y2) o None para todo el frame
        """
        # Configuración
        self.allowed_direction = allowed_direction
        self.detection_threshold = detection_threshold
        self.detection_zone = detection_zone
        
        # Diccionario de objetos trackados: id -> {posiciones, dirección, conteo wrong way}
        self.tracked_objects = {}
        
        # Lista de vehículos confirmados en sentido contrario
        self.wrong_way_vehicles = set()
        
        # Contador para IDs
        self.next_id = 0
        
        # Historial de detecciones para análisis
        self.detection_history = defaultdict(lambda: deque(maxlen=30))
        
        # Mapeo de direcciones a vectores de movimiento esperado
        self.direction_vectors = {
            'right': np.array([1, 0]),   # Movimiento hacia la derecha
            'left': np.array([-1, 0]),   # Movimiento hacia la izquierda
            'up': np.array([0, -1]),     # Movimiento hacia arriba
            'down': np.array([0, 1])     # Movimiento hacia abajo
        }
        
        # Parámetros específicos para la escena
        self.lane_divider = None  # Posición de la línea divisoria (si existe)
        self.min_movement = 15    # Movimiento mínimo para considerar que un vehículo se está moviendo
        
        print(f"Detector de sentido contrario inicializado - Dirección permitida: {allowed_direction}")
    
    def update(self, detections, frame_shape):
        """
        Actualiza el detector con nuevas detecciones de vehículos.
        
        Args:
            detections: Lista de tuplas (x1, y1, x2, y2, cls_id)
            frame_shape: Dimensiones del frame (h, w)
            
        Returns:
            Lista de tuplas (x1, y1, x2, y2, cls_id, object_id, is_wrong_way)
        """
        h, w = frame_shape[:2]
        current_time = time.time()
        
        # Usar la línea divisoria o calcularla si no está definida
        if self.lane_divider is None:
            self.lane_divider = w // 2  # Por defecto, usar el centro de la imagen
        
        # Lista de IDs detectados en este frame
        current_objects = []
        
        # Resultados: detecciones con flag de sentido contrario
        wrong_way_results = []
        
        # Procesar cada detección
        for detection in detections:
            x1, y1, x2, y2, cls_id = detection[:5]
            
            # Solo procesar vehículos (coche=2, camión=7, autobús=5, moto=3)
            if cls_id in [2, 3, 5, 7]:
                # Verificar si está en zona de detección
                if self.detection_zone:
                    zx1, zy1, zx2, zy2 = self.detection_zone
                    # Ignorar si está fuera de la zona
                    if x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2:
                        continue
                
                # Calcular centro del objeto
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_pos = np.array([center_x, center_y])
                
                # Intentar asociar con objetos previos (simple tracking por distancia)
                best_id = None
                min_dist = float('inf')
                
                for obj_id, obj_data in self.tracked_objects.items():
                    if 'positions' in obj_data and obj_data['positions']:
                        prev_pos = obj_data['positions'][-1]
                        dist = np.linalg.norm(current_pos - prev_pos)
                        # Umbral de distancia adaptativo basado en tamaño del objeto
                        size_factor = max(x2 - x1, y2 - y1) * 0.5
                        if dist < min_dist and dist < size_factor:
                            min_dist = dist
                            best_id = obj_id
                
                # Si no hay coincidencia, crear nuevo objeto
                if best_id is None:
                    best_id = self.next_id
                    self.next_id += 1
                    self.tracked_objects[best_id] = {
                        'first_seen': current_time,
                        'positions': [],
                        'direction': None,
                        'wrong_way_count': 0,
                        'class_id': cls_id,
                        'stationary_count': 0  # Contador para vehículos estacionados
                    }
                
                # Actualizar posiciones
                if 'positions' not in self.tracked_objects[best_id]:
                    self.tracked_objects[best_id]['positions'] = []
                
                self.tracked_objects[best_id]['positions'].append(current_pos)
                self.tracked_objects[best_id]['last_seen'] = current_time
                current_objects.append(best_id)
                
                # Limitar número de posiciones almacenadas
                if len(self.tracked_objects[best_id]['positions']) > 30:
                    self.tracked_objects[best_id]['positions'].pop(0)
                
                # ===== LÓGICA CORREGIDA PARA DETECCIÓN DE SENTIDO CONTRARIO =====
                
                # Valores por defecto
                movement_vector = np.array([0, 0])
                movement_distance = 0
                is_wrong_way = False
                
                # Verificar si tenemos suficientes posiciones para calcular dirección
                if len(self.tracked_objects[best_id]['positions']) >= 5:
                    # Uso de múltiples puntos para estabilizar la detección
                    start_pos = np.mean(self.tracked_objects[best_id]['positions'][:3], axis=0)
                    end_pos = np.mean(self.tracked_objects[best_id]['positions'][-3:], axis=0)
                    
                    # Vector y distancia de movimiento
                    movement_vector = end_pos - start_pos
                    movement_distance = np.linalg.norm(movement_vector)
                    
                    # Determinar si el vehículo está estacionado
                    if movement_distance < self.min_movement:
                        self.tracked_objects[best_id]['stationary_count'] += 1
                        # Vehículos estacionados nunca están en sentido contrario
                        if self.tracked_objects[best_id]['stationary_count'] > 5:
                            self.tracked_objects[best_id]['wrong_way_count'] = 0
                            if best_id in self.wrong_way_vehicles:
                                self.wrong_way_vehicles.remove(best_id)
                    else:
                        # Resetear contador de estacionado
                        self.tracked_objects[best_id]['stationary_count'] = 0
                        
                        # Normalizar vector de movimiento
                        if movement_distance > 0:
                            movement_vector = movement_vector / movement_distance
                            
                            # Determinar lado de la carretera (izquierdo o derecho)
                            is_left_side = center_x < self.lane_divider
                            
                            # Lógica específica para la escena - basado en la posición y dirección
                            if is_left_side:
                                # En el lado izquierdo:
                                # - Permitido: movimiento predominante hacia abajo o derecha
                                if movement_vector[1] < -0.5:  # Movimiento hacia arriba (contra)
                                    self.tracked_objects[best_id]['wrong_way_count'] += 1
                                elif movement_vector[0] < -0.5:  # Movimiento hacia izquierda (contra)
                                    self.tracked_objects[best_id]['wrong_way_count'] += 1
                                else:
                                    # Movimiento permitido
                                    self.tracked_objects[best_id]['wrong_way_count'] = max(
                                        0, self.tracked_objects[best_id]['wrong_way_count'] - 1
                                    )
                            else:
                                # En el lado derecho:
                                # - Permitido: movimiento predominante hacia arriba o izquierda
                                if movement_vector[1] > 0.5:  # Movimiento hacia abajo (contra)
                                    self.tracked_objects[best_id]['wrong_way_count'] += 1
                                elif movement_vector[0] > 0.5:  # Movimiento hacia derecha (contra) 
                                    self.tracked_objects[best_id]['wrong_way_count'] += 1
                                else:
                                    # Movimiento permitido
                                    self.tracked_objects[best_id]['wrong_way_count'] = max(
                                        0, self.tracked_objects[best_id]['wrong_way_count'] - 1
                                    )
                
                # Determinar si es wrong way confirmado (umbral alto para evitar falsos positivos)
                if self.tracked_objects[best_id]['wrong_way_count'] > self.detection_threshold:
                    self.wrong_way_vehicles.add(best_id)
                    is_wrong_way = True
                else:
                    # Quitar de la lista si ya no cumple el criterio
                    if best_id in self.wrong_way_vehicles:
                        self.wrong_way_vehicles.remove(best_id)
                    is_wrong_way = False
                    
                # Guardar la dirección de movimiento detectada
                if movement_distance > self.min_movement:
                    self.tracked_objects[best_id]['direction'] = movement_vector
                
                # Añadir a resultados con flag de sentido contrario y solo si no es estacionario
                is_stationary = self.tracked_objects[best_id].get('stationary_count', 0) > 5
                is_real_wrong_way = is_wrong_way and not is_stationary
                
                wrong_way_results.append((x1, y1, x2, y2, cls_id, best_id, is_real_wrong_way))
        
        # Limpiar objetos que no se han visto recientemente
        time_threshold = current_time - 3.0  # 3 segundos
        ids_to_remove = [
            obj_id for obj_id, obj_data in self.tracked_objects.items()
            if obj_data.get('last_seen', 0) < time_threshold
        ]
        for obj_id in ids_to_remove:
            if obj_id in self.tracked_objects:
                del self.tracked_objects[obj_id]
            if obj_id in self.wrong_way_vehicles:
                self.wrong_way_vehicles.remove(obj_id)
        
        return wrong_way_results
        
    def draw_results(self, frame, detections):
        """
        Dibuja los resultados con indicadores de sentido contrario.
        
        Args:
            frame: Imagen BGR
            detections: Lista de tuplas (x1, y1, x2, y2, cls_id, obj_id, is_wrong_way)
            
        Returns:
            Frame con anotaciones
        """
        annotated_frame = frame.copy()
        wrong_way_count = 0
        
        # Dibujar línea divisoria central
        h, w = frame.shape[:2]
        if self.lane_divider is None:
            self.lane_divider = w // 2
        
        # Dibujar línea divisoria (azul claro)
        cv2.line(annotated_frame, (self.lane_divider, 0), (self.lane_divider, h), 
                (255, 200, 0), 2, cv2.LINE_AA)
        
        # Dibujar flechas de dirección permitida en cada lado
        arrow_length = 40
        arrow_width = 30
        
        # Lado izquierdo: se permite bajar y a la derecha
        left_center = (self.lane_divider // 2, h // 2)
        # Flecha hacia abajo
        cv2.arrowedLine(annotated_frame, 
                      (left_center[0], left_center[1] - arrow_length),
                      (left_center[0], left_center[1] + arrow_length),
                      (0, 255, 255), 3, tipLength=0.3)
        # Flecha hacia la derecha
        cv2.arrowedLine(annotated_frame, 
                      (left_center[0] - arrow_length, left_center[1]),
                      (left_center[0] + arrow_length, left_center[1]),
                      (0, 255, 255), 3, tipLength=0.3)
        
        # Lado derecho: se permite subir y a la izquierda
        right_center = (self.lane_divider + (w - self.lane_divider) // 2, h // 2)
        # Flecha hacia arriba
        cv2.arrowedLine(annotated_frame, 
                      (right_center[0], right_center[1] + arrow_length),
                      (right_center[0], right_center[1] - arrow_length),
                      (0, 255, 255), 3, tipLength=0.3)
        # Flecha hacia la izquierda
        cv2.arrowedLine(annotated_frame, 
                      (right_center[0] + arrow_length, right_center[1]),
                      (right_center[0] - arrow_length, right_center[1]),
                      (0, 255, 255), 3, tipLength=0.3)
        
        # Dibujar detecciones
        for detection in detections:
            x1, y1, x2, y2, cls_id, obj_id, is_wrong_way = detection
            
            # Color según si está en sentido contrario o no
            if is_wrong_way:
                color = (0, 0, 255)  # Rojo para sentido contrario
                wrong_way_count += 1
            else:
                color = (0, 255, 0)  # Verde para dirección correcta
            
            # Dibujar recuadro
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Definir etiqueta
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
            
            # Añadir información de movimiento si está disponible
            is_stationary = False
            if obj_id in self.tracked_objects:
                if self.tracked_objects[obj_id].get('stationary_count', 0) > 5:
                    label += " [STATIONARY]"
                    is_stationary = True
                elif is_wrong_way:
                    label += " [WRONG WAY]"
            
            # Dibujar texto con fondo
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 10, y1),
                         color, -1)
            cv2.putText(annotated_frame, label, 
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Dibujar trayectoria si hay suficientes posiciones
            if obj_id in self.tracked_objects and len(self.tracked_objects[obj_id]['positions']) > 5:
                positions = self.tracked_objects[obj_id]['positions']
                
                # Dibujar línea de trayectoria
                pts = np.array([pos for pos in positions], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], False, color, 2)
                
                # Dibujar flecha de dirección si hay movimiento significativo
                if not is_stationary and len(positions) >= 5:
                    start_pos = positions[-5]
                    end_pos = positions[-1]
                    
                    if np.linalg.norm(end_pos - start_pos) > self.min_movement:
                        cv2.arrowedLine(annotated_frame, 
                                      tuple(start_pos.astype(int)), 
                                      tuple(end_pos.astype(int)), 
                                      color, 2, tipLength=0.5)
        
        # Si hay vehículos en sentido contrario, mostrar alerta
        if wrong_way_count > 0:
            cv2.rectangle(annotated_frame, 
                         (10, 10), 
                         (420, 70),
                         (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"¡ALERTA! {wrong_way_count} VEHÍCULOS EN SENTIDO CONTRARIO", 
                       (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_frame
    
    def set_lane_divider(self, position):
        """
        Establece la posición de la línea divisoria de carril.
        
        Args:
            position: Posición x de la línea divisoria
        """
        self.lane_divider = position
    
    def set_allowed_direction(self, new_direction):
        """
        Cambia la dirección permitida.
        
        Args:
            new_direction: Nueva dirección ('right', 'left', 'up', 'down')
        """
        if new_direction in self.direction_vectors:
            self.allowed_direction = new_direction
            # Reiniciar detecciones al cambiar dirección
            self.wrong_way_vehicles = set()
            for obj_id in self.tracked_objects:
                self.tracked_objects[obj_id]['wrong_way_count'] = 0
            return True
        return False
    
    def configure_scene(self, scene_type='intersection'):
        """
        Configura el detector para un tipo de escena específico.
        
        Args:
            scene_type: Tipo de escena ('intersection', 'highway', 'oneway')
        """
        if scene_type == 'intersection':
            self.detection_threshold = 8  # Más estricto en intersecciones
            self.min_movement = 15        # Exigir más movimiento para evitar falsos positivos
            # Reiniciar detecciones
            self.wrong_way_vehicles = set()
            for obj_id in self.tracked_objects:
                self.tracked_objects[obj_id]['wrong_way_count'] = 0
            return True
        return False