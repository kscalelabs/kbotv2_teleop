from setuptools import setup, find_packages

setup(
    name="vr_teleop",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "websockets",
        "pybullet",
        "numpy"
        # Add other dependencies here
    ],
)