from setuptools import setup, find_packages

setup(
    name="production-optimization",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask>=3.0",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "numpy>=1.24",
        "joblib>=1.3",
    ],
    author="Ing. Kelvin Cabrera",
    description="ML-based field production optimization and well allocation system",
    python_requires=">=3.9",
)
