import tkinter as tk
from src.app_manager import AppManager

def main():
    root = tk.Tk()
    root.title("InfractiVision")
    root.geometry("1280x720")
    # Inicia maximizada
    root.state("zoomed")
    AppManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
