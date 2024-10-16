from Detectify import *
from setuptools import setup, find_packages

setup(
    name="detectify-python",
    version="1.0.4",
    author="KANG CHANYOUNG",
    author_email="backgwa@icloud.com",
    description="Detectify - YOLOv11 based object detection framework",
    url="https://github.com/BackGwa/Detectify",
    install_requires=[
        "ultralytics==8.3.13"
    ],
    packages = find_packages(),
    python_requires=">=3.9.0"
)