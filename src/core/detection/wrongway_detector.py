import cv2
import numpy as np
import time
from collections import defaultdict, deque

class WrongWayDetector:
    def __init__(self, allowed_direction='right', detection_threshold=8, detection_zone=None):
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
        self.min_movement = 10    # Movimiento mínimo para considerar que un vehículo se está moviendo
        
        # Correcciones para la detección de sentido contrario
        self.manual_override = {}  # Para casos especiales donde queremos forzar el resultado
        self.debug_mode = False   # Para depuración
        
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
                        'stationary_count': 0,  # Contador para vehículos estacionados
                        'total_movement': 0     # Movimiento total acumulado
                    }
                
                # Actualizar posiciones
                if 'positions' not in self.tracked_objects[best_id]:
                    self.tracked_objects[best_id]['positions'] = []
                
                # Calcular movimiento desde la última posición
                if len(self.tracked_objects[best_id]['positions']) > 0:
                    prev_pos = self.tracked_objects[best_id]['positions'][-1]
                    movement = np.linalg.norm(current_pos - prev_pos)
                    self.tracked_objects[best_id]['total_movement'] += movement
                
                self.tracked_objects[best_id]['positions'].append(current_pos)
                self.tracked_objects[best_id]['last_seen'] = current_time
                current_objects.append(best_id)
                
                # Limitar número de posiciones almacenadas
                if len(self.tracked_objects[best_id]['positions']) > 30:
                    self.tracked_objects[best_id]['positions'].pop(0)
                
                # ===== LÓGICA MEJORADA PARA DETECCIÓN DE SENTIDO CONTRARIO =====
                
                # Por defecto, no es sentido contrario
                is_wrong_way = False
                is_stationary = False
                
                # Verificar si tenemos suficientes posiciones para calcular dirección
                if len(self.tracked_objects[best_id]['positions']) >= 4:
                    # Calcular basándonos en varias posiciones para mayor robustez
                    start_pos = np.mean(self.tracked_objects[best_id]['positions'][:2], axis=0)
                    end_pos = np.mean(self.tracked_objects[best_id]['positions'][-2:], axis=0)
                    
                    # Vector y distancia de movimiento
                    movement_vector = end_pos - start_pos
                    movement_distance = np.linalg.norm(movement_vector)
                    
                    # Determinar si el vehículo está estacionado
                    is_stationary = movement_distance < self.min_movement
                    
                    # Verificar movimiento significativo antes de determinar dirección
                    if movement_distance > self.min_movement:
                        # Resetear contador de estacionado
                        self.tracked_objects[best_id]['stationary_count'] = 0
                        
                        # Normalizar vector de movimiento
                        movement_vector = movement_vector / movement_distance
                        
                        # NUEVO: Comprobar si tenemos direcciones personalizadas definidas
                        use_custom_direction = False
                        
                        if hasattr(self, 'direction_vectors_custom') and self.direction_vectors_custom:
                            # Buscar la dirección personalizada que aplica a esta posición
                            applicable_direction = None
                            for dir_config in self.direction_vectors_custom:
                                region = dir_config['region']
                                if (region['x1'] <= center_x <= region['x2'] and 
                                    region['y1'] <= center_y <= region['y2']):
                                    applicable_direction = dir_config
                                    use_custom_direction = True
                                    break
                            
                            if applicable_direction:
                                # Calcular el ángulo entre el vector de movimiento y la dirección permitida
                                allowed_vector = applicable_direction['vector']
                                dot_product = np.dot(movement_vector, allowed_vector)
                                angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
                                angle_deg = np.degrees(angle_rad)
                                
                                # Si el ángulo es mayor a 90°, está yendo en dirección opuesta
                                if angle_deg > 90:
                                    self.tracked_objects[best_id]['wrong_way_count'] += 1
                                else:
                                    # Reducir el contador si se mueve en la dirección correcta
                                    self.tracked_objects[best_id]['wrong_way_count'] = max(
                                        0, self.tracked_objects[best_id]['wrong_way_count'] - 1
                                    )
                        
                        # Si no hay dirección personalizada aplicable, usar la lógica estándar
                        if not use_custom_direction:
                            # Determinar lado del carril (izquierdo o derecho de la línea azul)
                            is_left_side = center_x < self.lane_divider
                            
                            # Usar vector de dirección global configurado
                            allowed_vector = self.direction_vectors[self.allowed_direction]
                            
                            # Calcular ángulo entre movimiento y dirección permitida
                            dot_product = np.dot(movement_vector, allowed_vector)
                            
                            # Si el producto punto es negativo, el ángulo es mayor a 90 grados
                            if dot_product < 0:
                                self.tracked_objects[best_id]['wrong_way_count'] += 1
                            else:
                                self.tracked_objects[best_id]['wrong_way_count'] = max(
                                    0, self.tracked_objects[best_id]['wrong_way_count'] - 1
                                )
                    else:
                        # Incrementar contador de estacionado
                        self.tracked_objects[best_id]['stationary_count'] += 1
                        
                        # Los vehículos estacionados no están en sentido contrario
                        if self.tracked_objects[best_id]['stationary_count'] > 5:
                            is_stationary = True
                            # Resetear contadores para vehículos estacionados
                            self.tracked_objects[best_id]['wrong_way_count'] = 0
                            if best_id in self.wrong_way_vehicles:
                                self.wrong_way_vehicles.remove(best_id)
                
                # Confirmar sentido contrario con un alto umbral
                if self.tracked_objects[best_id]['wrong_way_count'] > self.detection_threshold:
                    # Asegurarse de que hay movimiento total suficiente para confirmar
                    if self.tracked_objects[best_id]['total_movement'] > self.min_movement * 3:
                        self.wrong_way_vehicles.add(best_id)
                        is_wrong_way = True
                else:
                    # Si ya no cumple el criterio, quitarlo de la lista
                    if best_id in self.wrong_way_vehicles:
                        self.wrong_way_vehicles.remove(best_id)
                
                # Añadir a resultados
                wrong_way_results.append((x1, y1, x2, y2, cls_id, best_id, is_wrong_way))
        
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
        
        # Dibujar línea divisoria central - COMENTADA PARA ELIMINAR LA LÍNEA CELESTE
        # h, w = frame.shape[:2]
        # if self.lane_divider is None:
        #     self.lane_divider = w // 2
        # cv2.line(annotated_frame, (self.lane_divider, 0), (self.lane_divider, h), 
        #        (255, 200, 0), 2, cv2.LINE_AA)
        
        # Dibujar flechas de dirección permitida - SOLO EN ESQUINAS, NO EN EL CENTRO
        h, w = frame.shape[:2]
        arrow_length = 30
        
        # Esquina inferior izquierda: flecha hacia abajo 
        cv2.arrowedLine(annotated_frame, 
                     (20, h-60), 
                     (20, h-20), 
                     (0, 255, 255), 2)
        
        # Esquina superior derecha: flecha hacia arriba
        cv2.arrowedLine(annotated_frame, 
                     (w-20, 60), 
                     (w-20, 20), 
                     (0, 255, 255), 2)
        
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
            
            # Añadir velocidad y estado si están disponibles
            if obj_id in self.tracked_objects:
                # Comprobar si está estacionado
                is_stationary = self.tracked_objects[obj_id].get('stationary_count', 0) > 5
                
                # Añadir etiqueta de estado
                if is_stationary:
                    # No añadimos [WRONG WAY] para vehículos estacionados
                    pass
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
        
        # Si hay vehículos en sentido contrario, mostrar alerta
        if wrong_way_count > 0:
            cv2.rectangle(annotated_frame, 
                         (10, 10), 
                         (420, 70),
                         (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"ALERTA: {wrong_way_count} VEHICULOS EN SENTIDO CONTRARIO", 
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
            self.detection_threshold = 12  # Más estricto en intersecciones
            self.min_movement = 15        # Exigir más movimiento para evitar falsos positivos
            # Reiniciar detecciones
            self.wrong_way_vehicles = set()
            for obj_id in self.tracked_objects:
                self.tracked_objects[obj_id]['wrong_way_count'] = 0
            return True
        return False
    
    def set_custom_directions(self, directions):
        """
        Establece direcciones personalizadas para el análisis de sentido contrario.
        
        Args:
            directions: Lista de diccionarios con 'start' y 'end' (coordenadas x,y)
                    que definen las direcciones permitidas de tráfico
        """
        self.custom_directions = directions
        
        # Calcular vectores de dirección normalizados para cada dirección personalizada
        self.direction_vectors_custom = []
        for dir_data in directions:
            start = np.array(dir_data['start'])
            end = np.array(dir_data['end'])
            
            # Vector dirección (normalizado)
            dir_vector = end - start
            dist = np.linalg.norm(dir_vector)
            
            if dist > 0:  # Evitar división por cero
                dir_vector = dir_vector / dist
                self.direction_vectors_custom.append({
                    'origin': start,
                    'vector': dir_vector,
                    'region': self._calculate_region(start, end)
                })
        
        # Reiniciar detecciones
        self.wrong_way_vehicles = set()
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['wrong_way_count'] = 0
        
        print(f"Configuradas {len(self.direction_vectors_custom)} direcciones personalizadas")

    def _calculate_region(self, start, end):
        """
        Calcula una región rectangular que contiene los puntos start y end.
        Utilizado para determinar qué regla de dirección aplicar a cada vehículo.
        """
        x1, y1 = start
        x2, y2 = end
        
        # Crear un rectángulo 10% más grande que la distancia entre los puntos
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Añadir margen (10% a cada lado)
        margin_x = width * 0.1
        margin_y = height * 0.1
        
        return {
            'x1': min_x - margin_x,
            'y1': min_y - margin_y,
            'x2': max_x + margin_x,
            'y2': max_y + margin_y
        }