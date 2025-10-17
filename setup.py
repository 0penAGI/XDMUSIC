from setuptools import setup, find_packages

setup(
    name="xdust",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "mido>=1.2.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A quantum-inspired avant-garde music generation library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xdust",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
