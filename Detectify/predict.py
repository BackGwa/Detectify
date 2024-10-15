import os
from ultralytics import YOLO
from .logger import Logger

class Predict:
    def __init__(self, model: str):
        self.log = Logger()
        
        if not os.path.exists(model):
            self.log.error(f"현재 경로에 '{model}'이 존재하지 않습니다.")

        try:
            self.log.alert(f"'{model}'을 불러오고 있습니다.")
            self.model = YOLO(model=model)
            self.log.success(f"'{model}'를 성공적으로 불러왔습니다.")
        except Exception as ex:
            self.log.error(f"'{model}'를 불러오던 중 문제가 발생했습니다.", ex)