"""Root-level build script — drives wheel building for PyPI and local
development installs (`pip install -e .`). BLAS/LAPACK linkage is picked
per platform:

    Linux   : -lopenblas -llapack (from manylinux yum packages)
    macOS   : -framework Accelerate
    Windows : OpenBLAS by default, or Intel MKL if AA_WINDOWS_BLAS=mkl.
              Paths read from OPENBLAS_ROOT (pre-built OpenBLAS install)
              or MKL_ROOT (conda-forge mkl-devel prefix).
"""
import os
import platform

from setuptools import Extension, setup
from Cython.Build import cythonize

# Keep enough metadata here for legacy setuptools code paths (`setup.py
# develop`, `setup.py egg_info`, older pip editable installs) to avoid
# synthesizing an `UNKNOWN` distribution. The canonical project metadata
# still lives in pyproject.toml.
NAME = "anderson-acceleration"
VERSION = "0.0.3"

include_dirs = ["include"]
library_dirs = []
libraries = []
extra_link_args = []

system = platform.system()
if system == "Darwin":
    extra_link_args = ["-framework", "Accelerate"]
elif system == "Windows":
    backend = os.environ.get("AA_WINDOWS_BLAS", "openblas").lower()
    if backend == "mkl":
        mkl_root = os.environ.get("MKL_ROOT")
        if mkl_root:
            include_dirs.append(os.path.join(mkl_root, "include"))
            library_dirs.append(os.path.join(mkl_root, "lib"))
        libraries = ["mkl_rt"]
    else:
        openblas_root = os.environ.get("OPENBLAS_ROOT")
        if openblas_root:
            include_dirs.append(os.path.join(openblas_root, "include"))
            library_dirs.append(os.path.join(openblas_root, "lib"))
        # OpenBLAS's Windows release zip names its MSVC import lib
        # `libopenblas.lib`, which matches `libraries=["libopenblas"]`
        # directly — no renaming needed.
        libraries = ["libopenblas"]
else:
    libraries = ["openblas", "lapack"]

aa_extension = Extension(
    name="aa",
    sources=["python/aapy.pyx"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_link_args=extra_link_args,
)

setup(
    name=NAME,
    version=VERSION,
    ext_modules=cythonize([aa_extension], language_level=3),
)
