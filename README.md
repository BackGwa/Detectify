<div align="center">

<br>

<img src="docs/images/detectify.png" width="400px"><br>

Detectify는 YOLO를 쉽고 빠르게 사용하기 위한, 라이브러리입니다.<br>

문제가 발생하거나, 질문이 있으시면, [Issue](https://github.com/BackGwa/Detectify/issues)를 남겨주세요. 최대한 도움을 드리겠습니다!<br>

해당 레포지토리 및 프로젝트에 기여를 하고싶다면, Fork해서 Pull-Request를 올려주세요!<br>
또한, 해당 프로젝트가 더 나아질 수 있도록, [토론](https://github.com/BackGwa/Detectify/discussions)을 시작할 수도 있습니다!

<br>

</div>

<br>

## 예제

- ### 모델 파인튜닝 (학습)
    #### 예제
    YOLOv11 모델을 커스텀 데이터셋으로 파인튜닝하는 과정입니다.<br>
    - **이 과정에선 GPU (CUDA)가 권장됩니다!**<br>
    *GPU가 없는 경우 CPU를 통해, 매우 느린 속도로 학습이 진행됩니다. (권장하지 않음!)*<br>
    *빠른 학습과 안정성을 위해, 해당 [Colab Notebook](https://colab.research.google.com/drive/1ZLOLCJn1IganamvmP8tKjUkyHjKzGAcs?usp=sharing)를 사용하여, 학습을 진행하세요.*
    
    - GPU 가속이 정상적으로 되지 않는 경우 [PyTorch](https://pytorch.org/)를 자신의 cuda 버전에 맞게 재설치해주세요.

    ```py
    from Detectify import Train

    train = Train()

    train.start(dataset="dataset/data.yaml")
    ```

- ### 모델 추론
    #### 예제
    학습된 모델을 사용하여, 추론하는 과정입니다.<br>
    - **이 과정에선 GPU (CUDA)가 권장됩니다!**<br>
    *GPU가 없는 경우 CPU를 통해, 느린 속도로 추론이 진행됩니다.*
    ```py
    from Detectify import Predict

    predict = Predict(model="model.pt")

    predict.start(source=0) # 추론 시작
    predict.stop()          # 추론 중지
    ```

    #### 값 가져오기
    추론에 관련한 값 (결괏값, 상태 등)을 가져올 수 있습니다.<br>
    ```py
    predict.status          # 추론 스레드가 시작되었는지 확인합니다.
                            # 자료형은 bool 입니다.

    predict.is_working      # 추론 중인지 확인합니다.
                            # 자료형은 bool 입니다.

    predict.result          # 추론 결괏값입니다.
                            # 자료형은 List[Results]이거나, None 입니다.

    predict.most            # 현재 최상위 클래스의 이름입니다.
                            # 자료형은 str 입니다.
    ```

    #### 추론 파라미터
    클래스의 파라미터를 조정하여, 추론 파라미터를 할 수 있습니다.<br>
    *추론 설정 과정은 되도록 추론 이전에 실행하는 것이 좋습니다.*
    ```py
    predict.preprocessing = None    # 가져온 소스의 프레임이 추론되기 전 거칠
                                    # 전처리 함수입니다. (하단 예제 참고)

    predict.handler = None  # 추론한 최상위 클래스가 변경되었을 때
                            # 핸들러가 실행할 함수입니다. (하단 예제 참고)
    
    predict.conf = 0.25     # 추론 시 감지 임계값입니다.
                            # 추론한 대상이 해당 임계값보다 미만이라면, 무시됩니다.
    
    predict.iou = 0.7       # 대상 중복 감지 임계값입니다.
                            # 값이 낮을수록 겹치는 영역이 적어집니다.

    predict.max_objs = 300  # 최대로 감지할 대상의 수 입니다.
                            # 하나의 물체만 감지해야한다면, 1로 설정하는 것이 좋습니다.
    ```

- ### 모델 추론 (전처리 예제)
    추론 전 이미지 전처리를 추가하는 예제입니다.<br>
    - **main 함수를 사용하는 것이 권장됩니다!**<br>
    *하단 예제는 main 함수 없이 작성되었지만, 이는 권장하지 않는 방법입니다.*
    ```py
    from Detectify import Predict

    predict = Predict("model.pt")

    def preprocessing(frame):
        # 이 곳에 이미지 전처리 코드를 삽입합니다...

    predict.preprocessing = preprocessing
    predict.start(source=0)
    ```

- ### 모델 추론 (핸들러 예제)
    #### 예제
    추론 과정에서 이벤트 핸들러를 추가하는 예제입니다.<br>
    이벤트 핸들러는 최상위 클래스가 변경될 때 호출됩니다.<br>
    - **main 함수를 사용하는 것이 권장됩니다!**<br>
    *하단 예제는 main 함수 없이 작성되었지만, 이는 권장하지 않는 방법입니다.*
    ```py
    from Detectify import Predict

    predict = Predict("model.pt")

    def callback():
        # 이 곳에 최상위 클래스가 변경되었을 때,
        # 작동할 코드를 삽입합니다...

    predict.handler = callback
    predict.start(source=0)
    ```
<br>

## 환경 구성

- ### 가상 환경으로 생성
    - Windows
     ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    pip install detectify-python
    ```

    - macOS / Linux
    ```bash
    python -m venv .venv
    source ./.venv/bin/activate

    pip install detectify-python
    ```

- ### Anaconda & Miniconda로 생성
    ```bash
    conda create -n Detectify python=3.11
    conda activate Detectify

    pip install detectify-python
    ```

---

## 기여자
- Keeworks 미래 광학기술 연구소 - [현장실습생 강찬영](https://github.com/BackGwa/)