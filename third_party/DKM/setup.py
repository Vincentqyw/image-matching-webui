from setuptools import setup, find_packages

setup(
    name="dkm",
    packages=find_packages(include=("dkm*",)),
    version="0.3.0",
    author="Johan Edstedt",
    install_requires=open("requirements.txt", "r").read().split("\n"),
)
