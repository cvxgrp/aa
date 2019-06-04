from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

aa_extension = Extension(
    name="aa",
    sources=["aapy.pyx"],
    library_dirs=["../src"],
    include_dirs=["../include"],
    libraries=['lapack', 'blas']
)

setup(
    name="aa",
    version='0.0.1',
    ext_modules=cythonize([aa_extension])
)

