# src/video/videoplayer_opencv.py

import cv2
import threading
import time
import queue
import tkinter as tk
import json
import os
import numpy as np
import torch
import psutil

from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
from ultralytics import YOLO

from src.core.detection.wrongway_detector import WrongWayDetector
from src.core.detection.speed_detector import SpeedDetector
from src.core.detection.plate_detector import PlateDetector
from src.core.detection.vehicle_detector import VehicleDetector
from src.core.processing.plate_processing import process_plate

# Archivos de configuración
POLYGON_CONFIG_FILE = "config/polygon_config.json"
AVENUE_CONFIG_FILE  = "config/avenue_config.json"
PRESETS_FILE        = "config/time_presets.json"

class VideoPlayerOpenCV:
    def __init__(self, parent, timestamp_updater, timestamp_label, semaforo):
        self.parent            = parent
        self.timestamp_updater = timestamp_updater
        self.timestamp_label   = timestamp_label
        self.semaforo          = semaforo

        self.seen_plates = set()
        self.wrong_way_detector = WrongWayDetector(allowed_direction='right')

        # Configuración CUDA para mejor rendimiento
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            
            # Configuraciones avanzadas para mejor rendimiento CUDA
            torch.backends.cudnn.benchmark = True     # Optimización para tamaños fijos de input
            torch.backends.cudnn.deterministic = False # Permite optimizaciones no deterministas
            torch.backends.cudnn.fastest = True       # Usa el algoritmo más rápido
            
            # Seleccionar GPU específica si tienes múltiples (opcional)
            # torch.cuda.set_device(0)  # Usa la primera GPU
            
            # Mostrar información de la GPU
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Usando GPU: {gpu_name} ({gpu_mem:.2f} GB)")
        else:
            self.device = torch.device('cpu')
            print("GPU no disponible, usando CPU")

        # Directorio de vídeos
        self.video_dir = os.path.join(os.getcwd(), "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        # Contenedor principal
        self.frame = tk.Frame(parent, bg='black')
        self.frame.pack(fill="both", expand=True)

        # Botonera inferior
        self.btn_frame = tk.Frame(self.frame, bg="black")
        self.btn_frame.pack(side="bottom", pady=12, anchor="w")

        btn_style = {
            "font": ("Arial", 12),
            "bg": "#4F4F4F",
            "fg": "white",
            "activebackground": "#6E6E6E",
            "activeforeground": "white",
            "bd": 0,
            "relief": "flat",
            "cursor": "hand2",
            "width": 14,
            "anchor": "center",
            "justify": "center"
        }

        self.load_button = tk.Button(
            self.btn_frame, text="CARGAR\nVIDEO",
            command=self.select_video,
            **btn_style
        )
        self.load_button.pack(side="left", padx=6)

        self.save_poly_button = tk.Button(
            self.btn_frame, text="GUARDAR\nÁREA",
            command=self.save_polygon,
            **btn_style
        )
        self.save_poly_button.pack(side="left", padx=6)

        self.delete_poly_button = tk.Button(
            self.btn_frame, text="BORRAR\nÁREA",
            command=self.delete_polygon,
            **btn_style
        )
        self.delete_poly_button.pack(side="left", padx=6)

        self.btn_gestion_polys = tk.Button(
            self.btn_frame, text="GESTIONAR\nÁREAS",
            command=self.gestionar_poligonos,
            **btn_style
        )
        self.btn_gestion_polys.pack(side="left", padx=6)

        self.btn_gestion_camaras = tk.Button(
            self.btn_frame, text="GESTIONAR\nCÁMARAS",
            command=self.gestionar_camaras,
            **btn_style
        )
        self.btn_gestion_camaras.pack(side="left", padx=6)


        # Panel vídeo + lateral
        self.video_panel_container = tk.Frame(self.frame, bg='black')
        self.video_panel_container.pack(side="top", fill="both", expand=True)

        self.video_frame = tk.Frame(
            self.video_panel_container, bg='black',
            width=640, height=360
        )
        self.video_frame.pack(side="left", fill="both", expand=True)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(
            self.video_frame, bg="black", bd=0, highlightthickness=0
        )
        self.video_label.pack(fill="both", expand=True)

        self.plates_frame = tk.Frame(
            self.video_panel_container, bg="gray", width=220
        )
        self.plates_frame.pack(side="right", fill="y")
        self.plates_frame.pack_propagate(False)

        self.plates_title = tk.Label(
            self.plates_frame, text="Placas Detectadas",
            bg="gray", fg="white", font=("Arial",16,"bold")
        )
        self.plates_title.pack(pady=10)

        self.plates_canvas = tk.Canvas(
            self.plates_frame, bg="gray", highlightthickness=0
        )
        self.plates_canvas.pack(side="left", fill="both", expand=True)

        self.plates_scrollbar = tk.Scrollbar(
            self.plates_frame, orient="vertical",
            command=self.plates_canvas.yview,
            bg="gray", troughcolor="gray", bd=0
        )
        self.plates_scrollbar.pack(side="right", fill="y")
        self.plates_canvas.configure(yscrollcommand=self.plates_scrollbar.set)

        self.plates_inner_frame = tk.Frame(self.plates_canvas, bg="gray")
        self.plates_inner_frame.bind(
            "<Configure>", self._on_plates_inner_configure
        )
        
        self.plates_canvas.create_window(
            (0,0), window=self.plates_inner_frame, anchor="nw"
        )
        self.detected_plates_widgets = []

        # Timestamp y avenida
        self.timestamp_label.config(
            font=("Arial",30,"bold"), bg="black", fg="yellow"
        )
        self.timestamp_label.place(in_=self.video_label, x=50, y=10)

        self.current_avenue = None
        self.avenue_label = tk.Label(
            self.video_frame, text="", font=("Arial",20,"bold"),
            bg="black", fg="white", wraplength=300
        )
        self.avenue_label.place(relx=0.5, y=80, anchor="n")

        # Info CPU/FPS/RAM
        self.info_label = tk.Label(
            self.video_frame, text="...", bg="black",
            fg="white", font=("Arial",11,"bold")
        )
        self.info_label.place(relx=0.98, y=10, anchor="ne")

        # Estado
        self.cap                = None
        self.running            = False
        self.orig_w, self.orig_h= None, None
        self.polygon_points     = []
        self.have_polygon       = False
        self.current_video_path = None

        # Cola acotada de OCR
        self.plate_queue   = queue.Queue(maxsize=1)
        self.plate_running = True
        self.plate_thread  = threading.Thread(
            target=self.plate_loop, daemon=True
        )
        self.plate_thread.start()

        # Métricas
        self.last_time = time.time()
        self.fps_calc  = 0.0
        self.using_gpu = torch.cuda.is_available()

        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(4)
        except:
            pass

        self.video_label.bind(
            "<Button-1>", self.on_mouse_click_polygon
        )

    def _on_plates_inner_configure(self, event):
        self.plates_canvas.configure(
            scrollregion=self.plates_canvas.bbox("all")
        )

    def load_avenue_config(self):
        if not os.path.exists(AVENUE_CONFIG_FILE):
            return {}
        try:
            with open(AVENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def save_avenue_config(self, data):
        with open(AVENUE_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_avenue_for_video(self, video_path):
        return self.load_avenue_config().get(video_path)

    def set_avenue_for_video(self, video_path, avenue_name):
        cfg = self.load_avenue_config()
        cfg[video_path] = avenue_name
        self.save_avenue_config(cfg)

    def load_time_presets(self):
        if not os.path.exists(PRESETS_FILE):
            return {}
        try:
            with open(PRESETS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}

    def save_time_presets(self, data):
        with open(PRESETS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def get_time_preset_for_video(self, video_path):
        return self.load_time_presets().get(video_path)

    def set_time_preset_for_video(self, video_path, times):
        presets = self.load_time_presets()
        presets[video_path] = times
        self.save_time_presets(presets)
        self.cycle_durations = times
        self.target_time     = time.time() + times[self.semaforo.get_current_state()]

    def first_time_setup(self, video_path):
        if ( self.get_avenue_for_video(video_path) is not None and
             self.get_time_preset_for_video(video_path) is not None ):
            messagebox.showinfo(
                "Info",
                "Este video ya fue configurado. Para abrirlo, use 'Gestionar Cámaras'.",
                parent=self.parent
            )
            return

        setup = tk.Toplevel(self.parent)
        setup.title("Configuración Inicial del Video")

        tk.Label(setup, text="Nombre de la Avenida:")\
          .grid(row=0, column=0, sticky="w", padx=5, pady=5)
        avenue_entry = tk.Entry(setup, width=30)
        avenue_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(setup, text="Tiempo Verde (s):")\
          .grid(row=1, column=0, sticky="w", padx=5, pady=5)
        green_entry = tk.Entry(setup, width=10)
        green_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(setup, text="Tiempo Amarillo (s):")\
          .grid(row=2, column=0, sticky="w", padx=5, pady=5)
        yellow_entry = tk.Entry(setup, width=10)
        yellow_entry.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(setup, text="Tiempo Rojo (s):")\
          .grid(row=3, column=0, sticky="w", padx=5, pady=5)
        red_entry = tk.Entry(setup, width=10)
        red_entry.grid(row=3, column=1, padx=5, pady=5)

        def guardar():
            ave = avenue_entry.get().strip()
            try:
                g = int(green_entry.get().strip())
                y = int(yellow_entry.get().strip())
                r = int(red_entry.get().strip())
            except ValueError:
                messagebox.showerror(
                    "Error", "Los tiempos deben ser enteros.", parent=setup
                )
                return
            if not ave:
                messagebox.showerror(
                    "Error", "Debe ingresar nombre de avenida.", parent=setup
                )
                return
            self.set_avenue_for_video(video_path, ave)
            self.current_avenue = ave
            self.avenue_label.config(text=ave)
            self.set_time_preset_for_video(video_path, {"green":g,"yellow":y,"red":r})
            messagebox.showinfo("Éxito","Configuración guardada.",parent=setup)
            setup.destroy()

        tk.Button(setup, text="Guardar Configuración", command=guardar)\
          .grid(row=4, column=0, columnspan=2,pady=10)

        setup.transient(self.parent)
        setup.grab_set()
        self.parent.wait_window(setup)

    def on_mouse_click_polygon(self, event):
        if self.have_polygon or self.orig_w is None:
            return
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl<2 or hlbl<2: return
        scale = min(wlbl/self.orig_w, hlbl/self.orig_h, 1.0)
        off_x = (wlbl - int(self.orig_w*scale))//2
        off_y = (hlbl - int(self.orig_h*scale))//2
        x_rel = (event.x - off_x)/scale
        y_rel = (event.y - off_y)/scale
        self.polygon_points.append((int(x_rel),int(y_rel)))

    def draw_polygon_on_np(self, img):
        if not self.polygon_points: return
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl<2 or hlbl<2: return
        scale = min(wlbl/self.orig_w, hlbl/self.orig_h, 1.0)
        off_x=(wlbl-int(self.orig_w*scale))//2
        off_y=(hlbl-int(self.orig_h*scale))//2
        pts_scaled=[(int(px*scale)+off_x,int(py*scale)+off_y)
                    for px,py in self.polygon_points]
        for i in range(len(pts_scaled)):
            x1,y1=pts_scaled[i]
            x2,y2=pts_scaled[(i+1)%len(pts_scaled)]
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    def save_polygon(self):
        if not self.cap or not self.current_video_path:
            messagebox.showerror("Error","No hay vídeo cargado.")
            return
        if len(self.polygon_points)<3:
            messagebox.showwarning("Advertencia","Al menos 3 vértices.")
            return
        self.have_polygon=True
        presets={}
        if os.path.exists(POLYGON_CONFIG_FILE):
            try:
                with open(POLYGON_CONFIG_FILE,"r",encoding="utf-8") as f:
                    presets=json.load(f)
            except: pass
        presets[self.current_video_path]=self.polygon_points
        with open(POLYGON_CONFIG_FILE,"w",encoding="utf-8") as f:
            json.dump(presets,f,indent=2)
        messagebox.showinfo("Éxito","Área guardada.")

    def load_polygon_for_video(self):
        self.have_polygon=False
        self.polygon_points=[]
        if not self.current_video_path or not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE,"r",encoding="utf-8") as f:
                presets=json.load(f)
            if self.current_video_path in presets:
                self.polygon_points=presets[self.current_video_path]
                self.have_polygon=True
        except: pass

    def delete_polygon(self):
        if not self.current_video_path or not self.polygon_points:
            messagebox.showwarning("Advertencia","No hay área.")
            return
        if not messagebox.askyesno("Confirmar","¿Borrar área?"):
            return
        try:
            with open(POLYGON_CONFIG_FILE,"r",encoding="utf-8") as f:
                presets=json.load(f)
            presets.pop(self.current_video_path,None)
            with open(POLYGON_CONFIG_FILE,"w",encoding="utf-8") as f:
                json.dump(presets,f,indent=2)
            self.have_polygon=False
            self.polygon_points=[]
            messagebox.showinfo("Éxito","Área eliminada.")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    def gestionar_poligonos(self):
        w = tk.Toplevel(self.parent)
        w.title("Áreas Guardadas")

        lb = tk.Listbox(w, width=80)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w, command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)

        # Cargar presets de áreas
        presets = {}
        if os.path.exists(POLYGON_CONFIG_FILE):
            try:
                with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                    presets = json.load(f)
            except Exception:
                presets = {}

        # Poblar listbox
        for video_path, points in presets.items():
            lb.insert(tk.END, f"{video_path} → {points}")

        # Botón de cierre
        tk.Button(w, text="Cerrar", command=w.destroy).pack(pady=5)

        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)


    def select_video(self):
        from tkinter import filedialog
        file = filedialog.askopenfilename(
            title="Seleccionar vídeo",
            filetypes=[("Vídeos","*.mp4 *.avi *.mov *.mkv"),("Todos","*.*")]
        )
        if not file:
            return
        fname = os.path.basename(file)
        dest  = os.path.join(self.video_dir, fname)
        if not os.path.exists(dest):
            import shutil
            shutil.copy2(file, dest)
        cap_tmp = cv2.VideoCapture(dest)
        cap_tmp.set(cv2.CAP_PROP_BUFFERSIZE,1)
        cap_tmp.release()
        self.stop_video()
        self.load_video(dest)

    def _load_video_async(self, path):
        cap_tmp = cv2.VideoCapture(path)
        cap_tmp.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            self.parent.after(0, lambda: messagebox.showerror("Error", "No se pudo leer el vídeo."))
            return
        self.parent.after(0, lambda: self._finish_loading_video(path, frame))

    def _finish_loading_video(self, path, first_frame):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.current_video_path = path
        h, w = first_frame.shape[:2]
        self.orig_h, self.orig_w = h, w
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_fps = max(self.cap.get(cv2.CAP_PROP_FPS), 30)
        self.running = True
        self.load_polygon_for_video()
        self.clear_detected_plates()
        self.semaforo.current_state = "green"
        self.semaforo.show_state()
        ave   = self.get_avenue_for_video(path)
        times = self.get_time_preset_for_video(path)
        if ave is None or times is None:
            self.first_time_setup(path)
        else:
            self.current_avenue = ave
            self.avenue_label.config(text=ave)
            self.cycle_durations = times
            self.target_time     = time.time() + times[self.semaforo.get_current_state()]
        if not self.timestamp_updater.running:
            self.timestamp_updater.start_timestamp()
        self.update_frames()

    def load_video(self, path):
        threading.Thread(
            target=self._load_video_async,
            args=(path,),
            daemon=True
        ).start()

    def stop_video(self):
        self.running = False
        if hasattr(self, "_after_id") and self._after_id:
            self.parent.after_cancel(self._after_id)
            self._after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None

    def plate_loop(self):
        while self.plate_running:
            try:
                frame, roi = self.plate_queue.get(timeout=1)
            except queue.Empty:
                continue
            bbox, plate_sr, ocr_text = process_plate(roi)
            if ocr_text:
                stamp = int(time.time())
                fname = f"plate_{ocr_text}_{stamp}.jpg"
                os.makedirs("data/output", exist_ok=True)
                cv2.imwrite(os.path.join("data/output", fname), plate_sr)
                self._safe_add_plate_to_panel(plate_sr, ocr_text)
            self.plate_queue.task_done()

    def detect_and_draw_cars(self, frame):
        """
        Detecta vehículos en el frame, estima su velocidad, identifica sentido contrario
        y muestra todo integrado en un único recuadro informativo.
        """
        frame_with_cars = frame.copy()
        car_detections = []
        
        try:
            # 1. Inicialización del detector de vehículos
            if not hasattr(self, 'vehicle_detector'):
                self.vehicle_detector = VehicleDetector(model_path="models/yolov8n.pt")
            
            # 2. Detectar vehículos (solo una detección para todo)
            detections = self.vehicle_detector.detect(frame, draw=False)
            
            # 3. Estimación de velocidad
            if not hasattr(self, 'speed_detector'):
                self.speed_detector = SpeedDetector(distance_meters=10, fps=30)
            
            if not hasattr(self, 'pixel_per_meter') or self.pixel_per_meter is None:
                self.pixel_per_meter = self.speed_detector.calibrate(detections)
            
            speed_detections = self.speed_detector.update(
                detections, frame.shape, self.pixel_per_meter
            )
            
            # 4. Detector de sentido contrario
            if not hasattr(self, 'wrong_way_detector'):
                self.wrong_way_detector = WrongWayDetector(
                    allowed_direction='right',
                    detection_threshold=10
                )
                # Configurar línea divisoria en el centro de la imagen
                h, w = frame.shape[:2]
                self.wrong_way_detector.set_lane_divider(w // 2)
                
            wrong_way_detections = self.wrong_way_detector.update(detections, frame.shape)
            
            # 5. Dibujar la línea divisoria y las flechas de dirección
            h, w = frame.shape[:2]
            lane_divider = w // 2
            # cv2.line(frame_with_cars, (lane_divider, 0), (lane_divider, h), 
            #         (255, 200, 0), 2, cv2.LINE_AA)
                    
            # Flechas de dirección permitida (pequeñas, solo en las esquinas)
            arrow_length = 30
            # Izquierda-abajo, derecha-arriba
            cv2.arrowedLine(frame_with_cars, 
                        (20, h-60), 
                        (20, h-20), 
                        (0, 255, 255), 2)
            cv2.arrowedLine(frame_with_cars, 
                        (w-20, 60), 
                        (w-20, 20), 
                        (0, 255, 255), 2)
            
            # 6. Integrar resultados: combinar velocidad y detección de sentido contrario
            vehicles_info = {}
            
            # Mapear por ID de objeto las detecciones de velocidad
            for detect in speed_detections:
                obj_id = detect[5]  # ID del objeto
                vehicles_info[obj_id] = {
                    'bbox': detect[:4],    # x1, y1, x2, y2
                    'cls_id': detect[4],   # clase
                    'speed': detect[6],    # velocidad
                    'wrong_way': False     # por defecto no está en sentido contrario
                }
                
            # Actualizar info con detecciones de sentido contrario
            for detect in wrong_way_detections:
                obj_id = detect[5]  # ID del objeto
                is_wrong_way = detect[6]  # Es sentido contrario
                
                if obj_id in vehicles_info:
                    vehicles_info[obj_id]['wrong_way'] = is_wrong_way
                else:
                    # Si por alguna razón no está en el diccionario
                    vehicles_info[obj_id] = {
                        'bbox': detect[:4],     # x1, y1, x2, y2
                        'cls_id': detect[4],    # clase
                        'speed': 0.0,           # velocidad desconocida
                        'wrong_way': is_wrong_way
                    }
                    
            # 7. Dibujar los resultados integrados
            for obj_id, info in vehicles_info.items():
                x1, y1, x2, y2 = info['bbox']
                cls_id = info['cls_id']
                speed = info['speed']
                wrong_way = info['wrong_way']
                
                # Color según si está en sentido contrario o no
                if wrong_way:
                    color = (0, 0, 255)  # Rojo para sentido contrario
                else:
                    # Color según velocidad
                    if speed < 30:
                        color = (0, 255, 0)  # Verde para velocidad baja
                    elif speed < 60:
                        color = (0, 255, 255)  # Amarillo para velocidad media
                    else:
                        color = (255, 165, 0)  # Naranja para velocidad alta
                
                # Dibujar recuadro
                cv2.rectangle(frame_with_cars, (x1, y1), (x2, y2), color, 2)
                
                # Texto de clase y velocidad
                if cls_id == 2:
                    class_text = "CAR"
                elif cls_id == 3:
                    class_text = "MOTORCYCLE"
                elif cls_id == 5:
                    class_text = "BUS"
                elif cls_id == 7:
                    class_text = "TRUCK"
                else:
                    class_text = f"CLASS {cls_id}"
                    
                # Crear etiqueta con formato solicitado: CAR: VELOCITY + WRONG WAY
                label = f"{class_text}: {speed:.1f} km/h"
                if wrong_way:
                    label += " [WRONG WAY]"
                    
                # Dibujar texto con fondo
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_with_cars, 
                            (x1, y1 - text_size[1] - 10), 
                            (x1 + text_size[0] + 10, y1),
                            color, -1)
                cv2.putText(frame_with_cars, label, 
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Añadir a formato de car_detections para el resto del código
                car_detections.append((x1, y1, x2, y2, cls_id, label))
                
            # 8. Alertas generales de sentido contrario
            wrong_way_count = sum(1 for info in vehicles_info.values() if info['wrong_way'])
            if wrong_way_count > 0:
                cv2.rectangle(frame_with_cars, 
                            (10, 10), 
                            (420, 70),
                            (0, 0, 255), -1)
                cv2.putText(frame_with_cars, f"ALERTA: {wrong_way_count} VEHICULOS EN SENTIDO CONTRARIO", 
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
        except Exception as e:
            print(f"Error al detectar vehículos y velocidad: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return frame_with_cars, car_detections

    def update_frames(self):
        """
        Actualiza los frames del video usando la función detect_and_draw_cars.
        """
        if not self.running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._after_id = self.parent.after(int(1000/30), self.update_frames)
            return

        # Usar nuestra nueva función para detectar y dibujar vehículos
        frame_with_cars, car_detections = self.detect_and_draw_cars(frame)
        
        # Si hay un polígono definido, dibujarlo
        if self.polygon_points:
            pts = np.array(self.polygon_points, np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame_with_cars, [pts], True, (0, 0, 255), 2)

        # Procesar placas si está en rojo (tu código existente)
        if self.semaforo.get_current_state() == "red" and not self.plate_queue.full():
            if self.polygon_points:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                roi = cv2.bitwise_and(frame, frame, mask=mask)
            else:
                roi = frame.copy()
            self.plate_queue.put((frame.copy(), roi))
        
        # Mostrar el frame anotado
        bgr_img = self.resize_and_letterbox(frame_with_cars)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        
        # Métricas y siguiente frame
        dt = time.time() - self.last_time
        self.last_time = time.time()
        if dt > 0:
            alpha = 0.9
            inst_fps = 1.0 / dt
            self.fps_calc = alpha * self.fps_calc + (1 - alpha) * inst_fps

        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        dev = "GPU" if self.using_gpu else "CPU"
        info_text = f"{dev} | FPS: {self.fps_calc:.1f} | RAM: {mem_mb:.1f}MB"
        self.info_label.config(text=info_text)
        
        # Asegurarse que las etiquetas estén visibles
        self.timestamp_label.lift()
        self.avenue_label.lift()
        self.info_label.lift()
        
        self._after_id = self.parent.after(10, self.update_frames)

    def _safe_add_plate_to_panel(self, plate_image, text):
        """
        Añade de forma segura una placa detectada al panel lateral,
        mostrando cada placa una debajo de otra de forma organizada.
        """
        # Evitar duplicados
        if text in self.seen_plates:
            return
        self.seen_plates.add(text)
        
        def _add():
            try:
                # 1. Redimensionar para que todas tengan el mismo tamaño
                h, w = plate_image.shape[:2]
                # Mantener relación de aspecto pero normalizar altura
                target_height = 60
                target_width = int(w * (target_height / h))
                
                # Limitar el ancho máximo al espacio disponible del panel
                max_width = 180
                if target_width > max_width:
                    target_width = max_width
                    target_height = int(h * (max_width / w))
                
                # Redimensionar con alta calidad
                thumb = cv2.resize(plate_image, (target_width, target_height), 
                                interpolation=cv2.INTER_AREA)
                
                # 2. Convertir a formato para Tkinter
                rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
                
                # 3. Crear contenedor para esta placa con fondo más oscuro para destacar
                plate_container = tk.Frame(self.plates_inner_frame, bg="#333333", 
                                        padx=5, pady=5, bd=1, relief="raised")
                
                # 4. Etiqueta para la imagen centrada
                lbl_img = tk.Label(plate_container, image=imgtk, bg="#333333")
                lbl_img.image = imgtk  # Mantener referencia
                lbl_img.pack(pady=(5, 3))
                
                # 5. Etiqueta para el texto en negritas para mejor legibilidad
                lbl_text = tk.Label(plate_container, text=text, bg="#333333", 
                                fg="white", font=("Arial", 12, "bold"))
                lbl_text.pack(pady=(0, 5))
                
                # 6. Empaquetar este contenedor en el panel principal con espaciado
                plate_container.pack(fill="x", pady=10, padx=10)
                
                # 7. Añadir a la lista para poder limpiarlos después
                self.detected_plates_widgets.append((plate_container, lbl_img, lbl_text))
                
                # 8. Ajustar scroll a la nueva altura
                self.plates_canvas.update_idletasks()
                self.plates_canvas.configure(scrollregion=self.plates_canvas.bbox("all"))
                
                # 9. Hacer scroll al final para mostrar el elemento recién añadido
                self.plates_canvas.yview_moveto(1.0)
                
            except Exception as e:
                print(f"Error al añadir placa al panel: {e}")
        
        # Ejecutar en el hilo principal de UI
        self.parent.after(0, _add)

    


    def resize_and_letterbox_cached(self, frame_bgr):
        """Versión optimizada con cache para resize_and_letterbox"""
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        
        # Si ya tenemos las dimensiones en cache, usar el cálculo previo
        if hasattr(self, 'letterbox_cache') and self.letterbox_cache['dims'] == (wlbl, hlbl, *frame_bgr.shape[:2]):
            scale = self.letterbox_cache['scale']
            new_w = self.letterbox_cache['new_w']
            new_h = self.letterbox_cache['new_h']
            off_x = self.letterbox_cache['off_x']
            off_y = self.letterbox_cache['off_y']
        else:
            # Calcular nuevos parámetros
            if wlbl < 2 or hlbl < 2:
                return frame_bgr
            
            h_ori, w_ori = frame_bgr.shape[:2]
            scale = min(wlbl / w_ori, hlbl / h_ori, 1.0)
            new_w = int(w_ori * scale)
            new_h = int(h_ori * scale)
            off_x = (wlbl - new_w) // 2
            off_y = (hlbl - new_h) // 2
            
            # Guardar en cache
            self.letterbox_cache = {
                'dims': (wlbl, hlbl, *frame_bgr.shape[:2]),
                'scale': scale,
                'new_w': new_w,
                'new_h': new_h,
                'off_x': off_x,
                'off_y': off_y
            }
        
        # Redimensionar frame con interpolación rápida
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crear canvas solo si es necesario
        if not hasattr(self, 'canvas_buffer') or self.canvas_buffer.shape[:2] != (hlbl, wlbl):
            self.canvas_buffer = np.zeros((hlbl, wlbl, 3), dtype=np.uint8)
        else:
            self.canvas_buffer.fill(0)  # Limpiar buffer
        
        # Colocar imagen redimensionada en el canvas
        self.canvas_buffer[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        
        return self.canvas_buffer

    def clear_detected_plates(self):
        for widget in self.detected_plates_widgets:
            widget.destroy()
        self.detected_plates_widgets.clear()
        self.seen_plates.clear()   # reinicia el set

    def gestionar_camaras(self):
        """
        Abre un diálogo para elegir un vídeo existente, y al 'Cargar'
        reinicia completamente el estado de Foto Rojo y carga el nuevo vídeo.
        """
        w = tk.Toplevel(self.parent)
        w.title("Gestionar Cámaras (videos)")

        lb = tk.Listbox(w, width=60)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(w, command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)

        for f in sorted(os.listdir(self.video_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                lb.insert(tk.END, f)

        btn_frame = tk.Frame(w)
        btn_frame.pack(fill="x", pady=5)

        def on_cargar():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Advertencia", "Seleccione un vídeo.")
                return
            fn   = lb.get(sel[0])
            path = os.path.join(self.video_dir, fn)
            w.destroy()

            # 1) Detener y limpiar todo el estado actual
            self.stop_video()
            self.clear_detected_plates()
            self.semaforo.current_state = "green"
            self.semaforo.show_state()

            # 2) Maximizar la ventana principal nuevamente
            main_win = self.parent.winfo_toplevel()
            main_win.deiconify()
            # main_win.state("zoomed")

            # 3) Cargar el nuevo vídeo
            self.load_video(path)

        tk.Button(btn_frame, text="Cargar",  width=10, command=on_cargar).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Borrar",  width=10, command=lambda: self._cam_del(lb)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cerrar",  width=10, command=w.destroy).pack(side="left", padx=5)

        w.transient(self.parent)
        w.grab_set()
        self.parent.wait_window(w)




    def _cam_load_async(self, path):
        cap_tmp = cv2.VideoCapture(path)
        cap_tmp.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, _ = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            self.parent.after(0, lambda: messagebox.showerror("Error", "No se pudo leer el vídeo."))
            return

        # Ahora volvemos al hilo principal:
        self.parent.after(0, lambda: (
            self.stop_video(),
            self.load_video(path)
        ))


    def _cam_load(self, lb):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia","Seleccione un vídeo.")
            return
        path = os.path.join(self.video_dir, lb.get(sel[0]))
        self.stop_video()
        self.load_video(path)

    def _cam_del(self, lb):
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Advertencia", "Seleccione un vídeo para borrar.")
            return
        fn = lb.get(sel[0])
        path = os.path.join(self.video_dir, fn)
        if not messagebox.askyesno("Confirmar", f"¿Borrar '{fn}'?"):
            return
        if path == self.current_video_path:
            self.running = False
            if hasattr(self, "_after_id") and self._after_id:
                self.parent.after_cancel(self._after_id)
                self._after_id = None
            if self.cap:
                self.cap.release()
                self.cap = None
            for item in self.detected_plates_widgets:
                item[0].destroy()
            self.detected_plates_widgets.clear()
            self.video_label.config(image="")
            self.current_video_path = None
        try:
            os.remove(path)
            self.remove_avenue_data(path)
            self.remove_time_preset_data(path)
            self.remove_polygon_data(path)
            lb.delete(sel[0])
            messagebox.showinfo("Info", f"'{fn}' y datos borrados.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def remove_video_data(self, video_path):
        self.remove_avenue_data(video_path)
        self.remove_time_preset_data(video_path)
        self.remove_polygon_data(video_path)

    def remove_avenue_data(self, video_path):
        if not os.path.exists(AVENUE_CONFIG_FILE):
            return
        try:
            with open(AVENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.pop(video_path, None)
            with open(AVENUE_CONFIG_FILE, "w", encoding="utf-8") as fw:
                json.dump(data, fw, indent=2)
        except:
            pass

    def remove_time_preset_data(self, video_path):
        if not os.path.exists(PRESETS_FILE):
            return
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
            presets.pop(video_path, None)
            with open(PRESETS_FILE, "w", encoding="utf-8") as fw:
                json.dump(presets, fw, indent=2)
        except:
            pass

    def remove_polygon_data(self, video_path):
        if not os.path.exists(POLYGON_CONFIG_FILE):
            return
        try:
            with open(POLYGON_CONFIG_FILE, "r", encoding="utf-8") as f:
                polygons = json.load(f)
            polygons.pop(video_path, None)
            with open(POLYGON_CONFIG_FILE, "w", encoding="utf-8") as fw:
                json.dump(polygons, fw, indent=2)
        except:
            pass

    def resize_and_letterbox(self, frame_bgr):
        wlbl = self.video_label.winfo_width()
        hlbl = self.video_label.winfo_height()
        if wlbl < 2 or hlbl < 2:
            return frame_bgr
        h_ori, w_ori = frame_bgr.shape[:2]
        scale = min(wlbl / w_ori, hlbl / h_ori, 1.0)
        new_w = int(w_ori * scale)
        new_h = int(h_ori * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((hlbl, wlbl, 3), dtype=np.uint8)
        off_x = (wlbl - new_w) // 2
        off_y = (hlbl - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas


# Fin del módulo VideoPlayerOpenCV
