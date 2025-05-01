import time
from datetime import datetime

class TimestampUpdater:
    def __init__(self, label, root):
        self.label = label
        self.root = root
        self.running = False

    def start_timestamp(self):
        self.running = True
        self.update()

    def stop_timestamp(self):
        self.running = False

    def update(self):
        if self.running:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.label.config(text=now_str)
            self.root.after(100, self.update)