# File: src/core/traffic_signal/semaphore.py

import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
import json
import os

# Ajustado a config folder
PRESETS_FILE = "config/time_presets.json"

class Semaforo:
    """
    Panel de semáforo:
    Ciclo simple: green -> yellow -> red, configurable mediante presets
    asociados a un nombre de vídeo.
    """

    def __init__(self, parent, visible=True):
        self.parent = parent
        self.visible = visible
        
        # Si no es visible, creamos un frame pero no lo empacamos
        self.frame = tk.Frame(parent, bg='white')
        if visible:
            self.frame.pack(side="top", fill="both", expand=True)

        # Canvas para semáforo
        self.canvas = tk.Canvas(self.frame, bg='white', highlightthickness=0)
        if visible:
            self.canvas.pack(fill="both", expand=True, pady=10)

        # Label de estado y tiempos
        self.info_label = tk.Label(self.frame, text="", font=("Arial", 14), bg='white')
        if visible:
            self.info_label.pack(pady=(0, 10))

        # Botón para abrir configuración de tiempos
        self.btn_tiempos = tk.Button(
            self.frame, text="Configurar Tiempos",
            command=self.gestionar_tiempos, width=20,
            bg="#3c3c3c", fg="white", bd=0, activebackground="#d9d9d9", activeforeground="white", pady=8,
        )
        if visible:
            self.btn_tiempos.pack(pady=5)

        # Dibujar carcasa y luces (incluso si no es visible, para mantener la lógica)
        self.housing_rect = self.canvas.create_rectangle(0, 0, 0, 0,
                                                        fill="black", outline="gray", width=3)
        self.red_light    = self.canvas.create_oval(0, 0, 0, 0, fill="grey", outline="white", width=1)
        self.yellow_light = self.canvas.create_oval(0, 0, 0, 0, fill="grey", outline="white", width=1)
        self.green_light  = self.canvas.create_oval(0, 0, 0, 0, fill="grey", outline="white", width=1)

        if visible:
            self.canvas.bind("<Configure>", self.resize_canvas)

        # Estado inicial y duraciones por defecto
        self.current_state = "green"
        self.cycle_durations = {"green": 30, "yellow": 3, "red": 30}
        self.target_time = time.time() + self.cycle_durations[self.current_state]

        self.show_state()
        self.update_countdown()

    # --------------------
    # Gestión de presets
    # --------------------
    def load_presets(self):
        if not os.path.exists(PRESETS_FILE):
            return {}
        try:
            with open(PRESETS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def save_presets(self, data):
        os.makedirs(os.path.dirname(PRESETS_FILE), exist_ok=True)
        with open(PRESETS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def gestionar_tiempos(self):
        """
        UI para listar todos los presets (clave = nombre de vídeo)
        y permitir agregar, editar o eliminar.
        """
        win = tk.Toplevel(self.parent)
        win.title("Configurar Tiempos - Vídeos")

        # Lista de presets
        tk.Label(win, text="Vídeos guardados:").grid(row=0, column=0, columnspan=3, pady=(5,0))
        lb = tk.Listbox(win, width=50)
        lb.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        # Scroll
        sb = tk.Scrollbar(win, orient="vertical", command=lb.yview)
        sb.grid(row=1, column=3, sticky="ns", pady=5)
        lb.config(yscrollcommand=sb.set)

        def refresh():
            lb.delete(0, tk.END)
            for vid, times in self.load_presets().items():
                g, y, r = times["green"], times["yellow"], times["red"]
                lb.insert(tk.END, f"{vid} → Verde={g}s, Amarillo={y}s, Rojo={r}s")

        refresh()

        # Entradas para nuevo/editar
        tk.Label(win, text="Nombre de vídeo:").grid(row=2, column=0, sticky="e", padx=5)
        entry_vid = tk.Entry(win, width=30)
        entry_vid.grid(row=2, column=1, columnspan=2, padx=5, pady=2)

        tk.Label(win, text="Verde (s):").grid(row=3, column=0, sticky="e", padx=5)
        entry_g = tk.Entry(win, width=6)
        entry_g.grid(row=3, column=1, sticky="w")

        tk.Label(win, text="Amarillo (s):").grid(row=4, column=0, sticky="e", padx=5)
        entry_y = tk.Entry(win, width=6)
        entry_y.grid(row=4, column=1, sticky="w")

        tk.Label(win, text="Rojo (s):").grid(row=5, column=0, sticky="e", padx=5)
        entry_r = tk.Entry(win, width=6)
        entry_r.grid(row=5, column=1, sticky="w")

        # Guardar preset (nuevo o editado)
        def on_save():
            vid = entry_vid.get().strip()
            try:
                g = int(entry_g.get().strip())
                y = int(entry_y.get().strip())
                r = int(entry_r.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Los tiempos deben ser números enteros.", parent=win)
                return
            if not vid:
                messagebox.showerror("Error", "Debe ingresar el nombre del vídeo.", parent=win)
                return
            presets = self.load_presets()
            presets[vid] = {"green": g, "yellow": y, "red": r}
            self.save_presets(presets)
            refresh()
            # Si editamos el preset activo, actualizar ciclo
            if vid == self.current_video:
                self.cycle_durations = presets[vid]
                self.target_time = time.time() + self.cycle_durations[self.current_state]
            messagebox.showinfo("Éxito", f"Tiempos guardados para '{vid}'.", parent=win)

        # Cargar selección en entries para editar
        def on_edit():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Advertencia", "Seleccione un ítem para editar.", parent=win)
                return
            line = lb.get(sel[0])
            vid, rest = line.split(" → ",1)
            times = self.load_presets().get(vid, {})
            entry_vid.delete(0, tk.END); entry_vid.insert(0, vid)
            entry_g.delete(0, tk.END); entry_g.insert(0, times.get("green",30))
            entry_y.delete(0, tk.END); entry_y.insert(0, times.get("yellow",3))
            entry_r.delete(0, tk.END); entry_r.insert(0, times.get("red",30))

        # Eliminar preset
        def on_delete():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Advertencia", "Seleccione un ítem para eliminar.", parent=win)
                return
            line = lb.get(sel[0])
            vid = line.split(" → ",1)[0]
            if messagebox.askyesno("Confirmar", f"Eliminar preset para '{vid}'?", parent=win):
                presets = self.load_presets()
                presets.pop(vid, None)
                self.save_presets(presets)
                refresh()

        tk.Button(win, text="Guardar", command=on_save).grid(row=6, column=0, pady=10)
        tk.Button(win, text="Cargar edición", command=on_edit).grid(row=6, column=1)
        tk.Button(win, text="Eliminar", command=on_delete).grid(row=6, column=2)

        win.transient(self.parent)
        win.grab_set()
        self.parent.wait_window(win)

    # --------------------
    # Ciclo de semáforo
    # --------------------
    def show_state(self):
        colors = {"green":self.green_light, "yellow":self.yellow_light, "red":self.red_light}
        for state, light in [("green",colors["green"]),
                             ("yellow",colors["yellow"]),
                             ("red",colors["red"])]:
            self.canvas.itemconfig(light, fill=state if state==self.current_state else "grey")

    def update_lights(self):
        # Rotar estado
        nxt = {"green":"yellow", "yellow":"red", "red":"green"}
        self.current_state = nxt[self.current_state]
        self.target_time = time.time() + self.cycle_durations[self.current_state]
        self.show_state()

    def update_countdown(self):
        now = time.time()
        diff = self.target_time - now
        if diff <= 0:
            self.update_lights()
            diff = self.target_time - time.time()
        secs = int(diff)
        ms = int((diff-secs)*1000)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.info_label.config(
            text=f"{ts}\nEstado: {self.current_state.upper()}"
        )
        self.frame.after(50, self.update_countdown)

    def get_current_state(self):
        return self.current_state

    def resize_canvas(self, event):
        cw, ch = event.width, event.height
        margin = 0.1 * min(cw, ch)
        max_w, max_h = int(cw-2*margin), int(ch-2*margin)
        hw = min(max_w, int(max_h*0.4))
        hh = int(hw/0.4)
        x0, y0 = (cw-hw)//2, (ch-hh)//2
        self.canvas.coords(self.housing_rect, x0, y0, x0+hw, y0+hh)
        sec = hh//3
        cx = x0+hw//2
        diam = min(int(0.8*hw), int(0.8*sec))
        for i, light in enumerate([self.red_light, self.yellow_light, self.green_light]):
            cy = y0 + sec//2 + i*sec
            self.canvas.coords(light,
                               cx-diam//2, cy-diam//2,
                               cx+diam//2, cy+diam//2)
