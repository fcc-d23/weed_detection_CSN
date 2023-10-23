from setuptools import setup
from Cython.Build import cythonize
import numpy
import cv2

setup(
    ext_modules = cythonize("frame_creation.pyx"),
     include_dirs=[numpy.get_include()]
)