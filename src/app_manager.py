# src/app_manager.py

import tkinter as tk
from src.gui.welcome_window import WelcomeFrame
from src.gui.red_light_violation_window import create_violation_window
from src.gui.infractions_management_window import create_infractions_window
import cv2
import numpy as np
from PIL import Image, ImageTk

class AppManager:
    """Centraliza la navegación entre pantallas GUI en una única ventana,
       preservando el estado (maximizado/minimizado) del root."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AnomalVision")
        # Arrancamos maximizado
        self.root.state("zoomed")
        # Mostramos bienvenida
        self.show_welcome()

    def _clear_root(self):
        """Destruye todos los widgets en root."""
        for w in self.root.winfo_children():
            w.destroy()

    def show_welcome(self):
        """Pantalla de bienvenida. No forza cambio de minimize/maximize."""
        prev_state = self.root.state()
        self._clear_root()
        self.root.title("AnomalVision – Principal")
        frm = WelcomeFrame(self.root, self)
        frm.pack(fill="both", expand=True)
        # Restauramos exactly el mismo estado (normal, iconic, zoomed)
        self.root.state(prev_state)

    def open_violation_window(self):
        """Pantalla de Foto Rojo."""
        prev_state = self.root.state()
        self._clear_root()
        self.root.title("AnomalVision – Detección de Placas")
        create_violation_window(self.root, self.show_welcome)
        self.root.state(prev_state)

    def open_infractions_window(self):
        """Pantalla de Gestión de Infracciones."""
        prev_state = self.root.state()
        self._clear_root()
        self.root.title("AnomalVision – Gestión de Infracciones")
        create_infractions_window(self.root, self.show_welcome)
        self.root.state(prev_state)
