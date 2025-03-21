from setuptools import setup, find_packages

setup(
    name="vr_teleop",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "websockets",
        "mujoco",
        "numpy",
        "pykos",
        "scipy"
        # Add other dependencies here
    ],
)