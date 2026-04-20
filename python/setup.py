from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

aa_extension = Extension(
    name="aa",
    sources=["aapy.pyx"],
    include_dirs=["../include"],
    libraries=['lapack', 'blas']
)

setup(
    name="aa",
    version='0.0.1',
    ext_modules=cythonize([aa_extension])
)
