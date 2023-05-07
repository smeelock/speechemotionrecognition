from setuptools import setup, find_packages

setup(
    name="speechemotionrecognition",
    description="An experimental project to understand speech signal representation for Speech Emotion Recognition (SER).",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.0.0",
    author="smeelock",
    license="MIT",
)
