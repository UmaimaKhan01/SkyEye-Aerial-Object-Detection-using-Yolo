from setuptools import setup, find_packages

setup(
    name="skyeye_detection",
    version="0.1.0",
    description="SkyEye: Transformer-Based Object Detection for Aerial Imagery",
    author="Umaima Khan, Ajay Ramesh, Sai Nikhil Chidipothu, Mani Sai Gandra",
    author_email="fn653419@ucf.edu, aj608325@ucf.edu, sa419655@ucf.edu, ma240368@ucf.edu",
    url="https://github.com/UmaimaKhan01/skyeye-aerial-object-detection-using-Yolo",  
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "numpy>=1.18.5",
        "opencv-python>=4.1.2",
        "matplotlib>=3.2.2",
        "PyYAML>=5.3.1",
        "tqdm>=4.41.0",
        "pandas>=1.1.4",
        "seaborn>=0.11.0",
        "tensorboard>=2.4.1",
        "Pillow>=7.1.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=20.8b1",
            "isort>=5.7.0",
            "flake8>=3.8.4",
        ],
    },
)
