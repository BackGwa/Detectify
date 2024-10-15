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
    학습된 모델을 사용하여, 추론하는 과정입니다.<br>
    - **이 과정에선 GPU (CUDA)가 권장됩니다!**<br>
    *GPU가 없는 경우 CPU를 통해, 느린 속도로 추론이 진행됩니다.*
    ```py
    from Detectify import Predict

    predict = Predict(model="model.pt")
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