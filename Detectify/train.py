import os, requests, yaml
from pathlib import Path
from ultralytics import YOLO
from .util import ask, get_device
from .logger import Logger


class Train:
    def __init__(self, model: str = "yolo11n.pt"):
        """
        Detectify에서 YOLOv11 기반의 모델 파인튜닝 인스턴스를 생성합니다.

        Args:
            model (str, optional):
                파인튜닝 할 모델의 경로를 설정합니다.
                *기본 값은 `yolo11n.pt` 입니다.*

                - `yolo11n.pt`: YOLOv11 nano *(39.5 mAP)*
                - `yolo11s.pt`: YOLOv11 small *(47.0 mAP)*
                - `yolo11m.pt`: YOLOv11 medium *(51.5 mAP)*
                - `yolo11l.pt`: YOLOv11 large *(53.4 mAP)*
                - `yolo11x.pt`: YOLOv11 extra large *(54.7 mAP)*

        """
        self.log = Logger()

        try:
            while True:
                if os.path.exists(model):
                    self.log.alert(f"'{model}'을 불러오고 있습니다.")
                    self.model = YOLO(model=model)
                    self.log.success(f"'{model}'를 성공적으로 불러왔습니다.")
                    break
                else:
                    self.log.warn(f"현재 경로에 '{model}'이 존재하지 않습니다.")
                    if ask("모델을 다운로드하시겠습니까?"):
                        link = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model}"
                        try:
                            file = requests.get(link)
                            if file.status_code == 404:
                                raise ValueError(f"'{model}'은 유효하지 않은 모델 이름입니다.")
                            open(model, 'wb').write(file.content)
                            self.log.success(f"'{model}' 다운로드를 성공했습니다.")
                        except Exception as ex:
                            self.log.error(f"'{model}' 다운로드를 실패하였습니다.", ex)
                    else:
                        raise Exception("모델 다운로드를 취소하여, 인스턴스 생성에 실패하였습니다.")
        except Exception as ex:
            self.log.error(f"'{model}'를 불러오던 중 문제가 발생했습니다.", ex)

    def start(self, dataset: str, output: str = "output", epochs: int = 100, batch_size: int = 16, save_per_epochs: int = 10, cache: bool = False, resume: bool = False):
        """
        Detectify에서 YOLOv11 기반의 모델을 파인튜닝합니다.

        Args:
            dataset (str):
                데이터셋 파일을 선택합니다.
                `.yaml` 파일 내 값이 모두 유효해야합니다.
            output (str):
                모델의 학습 결과를 저장할 경로를 설정합니다.\n
                *기본 값은 `output` 입니다.*
            epochs (int, optional):
                학습 에포크 수를 설정합니다.
                모델의 정확도 및 안정성을 고려하여, 조정하세요.\n
                *기본 값은 `100` 입니다.*
            batch_size (int, optional):
                학습 배치 사이즈를 설정합니다.
                시스템의 성능을 고려하여, 조정하세요.\n
                *기본 값은 `16` 입니다.*
            save_per_epochs (int, optional):
                모델을 설정 에포크마다 저장합니다.
                에포크 수와 시스템의 저장용량을 고려하여, 조정하세요.\n
                *기본 값은 `25` 입니다.
            cache (bool, optional):
                메모리를 캐시로 활용합니다.
                시스템 메모리를 고려하여, 조정하세요.\n
                *기본 값은 `False` 입니다.*
            resume (bool, optional):
                학습을 이어서 시작합니다.\n
                *기본 값은 `False` 입니다.*
        """

        try:
            self.log.alert(f"데이터셋 유효성 검사를 시작합니다.")
            
            if not os.path.exists(dataset):
                raise FileNotFoundError("데이터셋이 유효하지 않습니다. 학습을 종료합니다.")

            with open(dataset) as file:
                config = yaml.full_load(file)

            for key, path in (("train", "train"), ("val", "valid"), ("test", "test")):
                if not os.path.exists(config[key]):
                    self.log.warn(f"{dataset} 내의 {key}의 경로를 찾을 수 없습니다.")
                    if ask("자동 수정을 진행하시겠습니까?"):
                        dir = Path(dataset).parent.absolute()
                        fix_path = os.path.join(dir, path, "images")
                        config[key] = fix_path

                        with open(dataset, 'w') as file:
                            yaml.dump(config, file)
                    else:
                        raise Exception("데이터 자동수정을 진행하지 않아, 중단되었습니다.")
                else:
                    self.log.success(f"{dataset} 내의 {key} 데이터셋 유효성 검사를 통과했습니다.")
        except Exception as ex:
            self.log.error("데이터셋 유효성 검사를 실패하였습니다.", ex)

        device = get_device()

        try:
            self.log.alert(f"파인튜닝을 시작합니다.")
            self.model.train(
                data=dataset,
                epochs=epochs,
                batch=batch_size,
                save_period=save_per_epochs,
                cache=cache,
                project=output,
                device=device,
                resume=resume,
                workers=0,
                patience=0
            )
            self.log.alert(f"파인튜닝이 완료되었습니다!")
        except Exception as ex:
            self.log.error("파인튜닝을 진행하던 중, 문제가 발생했습니다.", ex)