import tkinter as tk
from tkinter import messagebox
import json, os
from tkcalendar import DateEntry
from datetime import datetime

INF_FILE = os.path.join("data", "infracciones.json")

def create_infractions_window(window: tk.Toplevel, back_callback):
    window.configure(bg="#ffffff")
    window.state("zoomed")

    # — Header con botón volver, título y acciones —
    header = tk.Frame(window, bg="#ffffff")
    header.pack(fill="x", padx=30, pady=20)

    tk.Button(
        header, text="Volver", font=("Arial", 16), bg="#3c3c3c", fg="white",
        bd=0, activebackground="#d9d9d9", activeforeground="white",
        command=back_callback, cursor="hand2"
    ).pack(side="left")

    tk.Label(
        header, text="Gestión de Métricas",
        font=("Arial", 28, "bold"), bg="#ffffff", fg="black"
    ).pack(side="left", padx=(20,0))

    actions = tk.Frame(header, bg="#ffffff")
    actions.pack(side="right")

    tk.Button(
        actions, text="DESCARGAR", font=("Arial", 14),
        bg="#3c3c3c", fg="white", bd=0,
        activebackground="#d9c9d9", activeforeground="white",
        cursor="hand2"
    ).pack(side="left", padx=10)

    # Date range pickers
    tk.Label(actions, text="Desde:", font=("Arial", 12), bg="#ffffff").pack(side="left")
    start_picker = DateEntry(
        actions, font=("Arial", 12), width=10,
        background="white", foreground="black",
        borderwidth=1, date_pattern='dd/MM/yyyy'
    )
    start_picker.pack(side="left", padx=(5,15))

    tk.Label(actions, text="Hasta:", font=("Arial", 12), bg="#ffffff").pack(side="left")
    end_picker = DateEntry(
        actions, font=("Arial", 12), width=10,
        background="white", foreground="black",
        borderwidth=1, date_pattern='dd/MM/yyyy'
    )
    end_picker.pack(side="left", padx=(5,15))

    def apply_filter():
        try:
            start = datetime.combine(start_picker.get_date(), datetime.min.time())
            end = datetime.combine(end_picker.get_date(), datetime.max.time())
            filtered = []
            for inf in all_data:
                fecha_str = inf.get('fecha','')
                fecha = datetime.strptime(fecha_str, '%d/%m/%Y')
                if start <= fecha <= end:
                    filtered.append(inf)
            populate_cards(filtered)
            update_results_summary(filtered)
        except Exception as e:
            messagebox.showerror("Error", f"Error aplicando filtro: {e}")

    tk.Button(
        actions, text="FILTRAR", font=("Arial", 12),
        bg="#3c3c3c", fg="white", bd=0,
        activebackground="#d9c9d9", activeforeground="white",
        cursor="hand2", command=apply_filter
    ).pack(side="left", padx=10)

    # — Sección de resultados —
    results_frame = tk.Frame(window, bg="#f9f9f9")
    results_frame.pack(fill="x", padx=30, pady=(30,20))
    tk.Label(
        results_frame, text="Resultados:", 
        font=("Arial", 18, "bold"), bg="#f9f9f9", fg="#333333"
    ).pack(side="left", padx=(0,20))
    total_label = tk.Label(
        results_frame, text="Total de registros: 0",
        font=("Arial", 14), bg="#f9f9f9", fg="#555555"
    )
    total_label.pack(side="left", padx=(0,20))
    range_label = tk.Label(
        results_frame, text="Rango: --/--/---- - --/--/----",
        font=("Arial", 14), bg="#f9f9f9", fg="#555555"
    )
    range_label.pack(side="left", padx=(0,20))
    unique_label = tk.Label(
        results_frame, text="Placas únicas: 0",
        font=("Arial", 14), bg="#f9f9f9", fg="#555555"
    )
    unique_label.pack(side="left", padx=(0,20))
    models_label = tk.Label(
        results_frame, text="Modelos únicos: 0",
        font=("Arial", 14), bg="#f9f9f9", fg="#555555"
    )
    models_label.pack(side="left")

    def update_results_summary(data_list):
        total_label.config(text=f"Total de registros: {len(data_list)}")
        # Actualiza rango basado en los pickers
        start_txt = start_picker.get_date().strftime('%d/%m/%Y')
        end_txt = end_picker.get_date().strftime('%d/%m/%Y')
        range_label.config(text=f"Rango: {start_txt} - {end_txt}")
        # Calcula placas únicas
        placas = {inf.get('placa','') for inf in data_list}
        unique_label.config(text=f"Placas únicas: {len(placas)}")
        # Calcula modelos únicos
        modelos = {inf.get('modelo','') for inf in data_list}
        models_label.config(text=f"Modelos únicos: {len(modelos)}")

    # — Contenedor scrollable para las tarjetas —
    container = tk.Frame(window, bg="gray")
    container.pack(fill="both", expand=True, padx=160, pady=20)

    canvas = tk.Canvas(container, bg="gray", highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="gray")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Load all data once
    if os.path.exists(INF_FILE):
        try:
            with open(INF_FILE, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando métricas: {e}")
            all_data = []
    else:
        all_data = []

    def clear_cards():
        for child in scrollable_frame.winfo_children():
            child.destroy()

    def populate_cards(data_list):
        clear_cards()
        update_results_summary(data_list)
        if not data_list:
            tk.Label(
                scrollable_frame, text="No se encontraron métricas.",
                font=("Arial", 16), bg="gray", fg="white"
            ).pack(pady=80, padx=80)
            return
        for inf in data_list:
            card = tk.Frame(scrollable_frame, bg="#F2F2F2")
            card.pack(fill="x", padx=20, pady=10)

            img_frame = tk.Frame(card, width=120, height=80, bg="black")
            img_frame.pack(side="left", padx=10, pady=10)
            img_frame.pack_propagate(False)

            text_left = tk.Frame(card, bg="#F2F2F2")
            text_left.pack(side="left", fill="y", padx=(0,20), pady=10)
            tk.Label(
                text_left, text=f"Modelo: {inf.get('modelo','N/A')}",
                font=("Arial", 12), bg="#F2F2F2", fg="#333333"
            ).pack(anchor="w")
            tk.Label(
                text_left, text=f"Color: {inf.get('color','N/A')}",
                font=("Arial", 12), bg="#F2F2F2", fg="#333333"
            ).pack(anchor="w")
            tk.Label(
                text_left, text=f"Placa: {inf.get('placa','N/A')}",
                font=("Arial", 12, "bold"), bg="#F2F2F2", fg="#273D86"
            ).pack(anchor="w")
            tk.Label(
                text_left, text=f"Fecha: {inf.get('fecha','')}   Hora: {inf.get('hora','')}",
                font=("Arial", 12), bg="#F2F2F2", fg="#555555"
            ).pack(anchor="w")

            tk.Frame(card, bg="#CCCCCC", width=2).pack(side="left", fill="y", pady=10)

            text_right = tk.Frame(card, bg="#F2F2F2")
            text_right.pack(side="left", fill="both", expand=True, padx=20, pady=10)
            tk.Label(
                text_right, text=f"Ubicación: {inf.get('ubicacion','Desconocida')}",
                font=("Arial", 12), bg="#F2F2F2", fg="#333333",
                wraplength=300, justify="left"
            ).pack(anchor="w")
            tk.Label(
                text_right, text=f"Coordenadas: {inf.get('coordenadas','')}",
                font=("Arial", 12), bg="#F2F2F2", fg="#333333"
            ).pack(anchor="w")

    # initially show all
    populate_cards(all_data)
