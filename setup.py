from setuptools import setup, find_packages

setup(
    name="mmda-project",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "open-clip-torch>=2.24.0",
        "numpy",
        "pillow",
        "omegaconf",
        "pyyaml",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pandas",
        "tqdm",
        "tensorboard",
    ],
    extras_require={
        "dev": ["pytest", "umap-learn"],
    },
)
