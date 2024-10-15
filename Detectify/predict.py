from ultralytics import YOLO

class Predict:
    def __init__(self, model: str):
        self.model = YOLO(model=model)