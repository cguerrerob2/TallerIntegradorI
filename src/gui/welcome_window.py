# src/gui/welcome_window.py

import tkinter as tk
import os
from PIL import Image, ImageTk

class WelcomeFrame(tk.Frame):
    def __init__(self, master, app_manager):
        super().__init__(master, bg="#3c3c3c")
        self.app_manager = app_manager
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.create_widgets()

    def create_widgets(self):
        # Panel izquierdo: imagen
        left = tk.Frame(self, bg="#3c3c3c")
        left.grid(row=0, column=0, sticky="nsew")
        bg_path = os.path.join("img", "welcome_bg.png")
        try:
            img_orig = Image.open(bg_path)
            lbl = tk.Label(left)
            lbl.place(relwidth=1, relheight=1)
            def resize(e):
                img = img_orig.resize((e.width, e.height), Image.Resampling.LANCZOS)
                self._tk = ImageTk.PhotoImage(img)
                lbl.config(image=self._tk)
            left.bind("<Configure>", resize)
        except:
            tk.Label(left, text="[Imagen no disponible]", bg="#3c3c3c", fg="white").pack(expand=True)

        # Panel derecho: títulos + botones
        right = tk.Frame(self, bg="white")
        right.grid(row=0, column=1, sticky="nsew")
        content = tk.Frame(right, bg="white")
        content.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(content, text="Bienvenido a\nAnomalVision",
                 font=("Arial", 40, "bold"), bg="white", fg="#3c3c3c",
                 justify="center").pack(pady=(0,10))

        tk.Label(content, text="Selecciona la opción para continuar",
                 font=("Arial", 18), bg="white", fg="gray20",
                 justify="center").pack(pady=(0,30))

        btns = tk.Frame(content, bg="white")
        btns.pack()
        def mk(txt, cmd):
            return tk.Button(btns, text=txt, font=("Arial",16),
                             bg="#3c3c3c", fg="white",
                             activebackground="#d9d9d9", bd=0,
                             padx=20, pady=10, cursor="hand2",
                             command=cmd)
        # FOTO ROJO primero a la izquierda
        mk("Panel de Monitoreo", self.app_manager.open_violation_window).pack(side="left", padx=10)
        mk("Gestión de Metricas", self.app_manager.open_infractions_window).pack(side="left", padx=10)
