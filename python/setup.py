from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

examples_extension = Extension(
    name="aa",
    sources=["aapy.pyx"],
    library_dirs=["../src"],
    include_dirs=["../include"],
    libraries=['lapack', 'blas']
)
setup(
    name="aa",
    ext_modules=cythonize([examples_extension])
)

