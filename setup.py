from Detectify import *
from setuptools import setup, find_packages

setup(
    name="Detectify",
    version="1.0.0",
    author="KANG CHANYOUNG",
    author_email="backgwa@icloud.com",
    description="Detectify",
    url="https://github.com/BackGwa/Detectify",
    install_requires=[
        "ultralytics==8.3.13"
    ],
    packages = find_packages(),
    python_requires=">=3.11.10"
)