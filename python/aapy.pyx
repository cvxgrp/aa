import numpy as np

cdef extern from "../src/aa.c":
    pass

cdef extern from "../include/aa.h":
    ctypedef struct AaWork:
        pass
    AaWork *aa_init(int, int, int, int, float)
    int aa_apply(double*, const double*, AaWork*)
    void aa_finish(AaWork *)

cdef class AndersonAccelerator(object):
    cdef AaWork* _wrk
    def __cinit__(self, dim, mem, type1=False, interval=1, eta=1e-9):
        self._wrk = aa_init(dim, mem, type1, interval, eta)

    def apply(self, f, x):
        f = np.squeeze(f)
        x = np.squeeze(x)
        if not f.flags['C_CONTIGUOUS']:
            f = np.ascontiguousarray(f) # Makes a contiguous copy of the numpy array.
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x) # Makes a contiguous copy of the numpy array.

        cdef double[::1] f_memview = f
        cdef double[::1] x_memview = x

        success = aa_apply(&f_memview[0], &x_memview[0], self._wrk)

        return success

    def __dealloc__(self):
        aa_finish(self._wrk)

