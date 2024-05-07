from setuptools import find_packages
from distutils.core import setup

setup(
    name="zzs_legged_gym",
    version="1.0.0",
    author="Nikita Rudin",
    maintainer="zishun zhou",
    maintainer_email="zhouzishun@mail.zzshub.cn",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="rudinn@ethz.ch",
    description="ZZS version Isaac Gym environments for Legged Robots",
    install_requires=[
        "isaacgym",
        "torch>=1.4.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "matplotlib",
        "tensorboard>=1.15",
        "onnx",
    ],
)
