from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy as np

# get np include path
np_include_path = np.get_include()

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    "dev_fn.external.libmesh.triangle_hash",
    sources=["src/dev_fn/external/libmesh/triangle_hash.pyx"],
    libraries=["m"],  # Unix-like specific
    include_dirs=[np_include_path],
)

# Gather all extension modules
cython_ext_modules = [
    triangle_hash_module,
]
ext_modules = cythonize(cython_ext_modules)

setup(
    name="oakink2_tamf",
    version="0.0.1",
    python_requires=">=3.9.0",
    packages=find_packages(
        where="src",
        include="oakink2_tamf*",
    ),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
