from setuptools import setup, find_packages

setup(
    name="motion-control",
    version="1.0.0",
    description="Animation faciale par transfert de mouvement : applique les mouvements d'une vidéo à une image de référence",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "customtkinter==5.2.2",
        "Pillow==10.4.0",
        "opencv-python==4.10.0.84",
        "mediapipe==0.10.14",
        "numpy==1.26.4",
    ],
)
