from setuptools import setup, find_packages
# from distutils.extension import Extension
# from Cython.Build import cythonize
# from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
# import numpy as np

setup(
    name="oakink2_tamf",
    version="0.0.1",
    python_requires=">=3.9.0",
    packages=find_packages(
        where="src",
        include="oakink2_tamf*",
    ),
    package_dir={"": "src"},
)
