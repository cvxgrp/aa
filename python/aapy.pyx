import numpy as np

cdef extern from "../src/aa.c":
    pass

cdef extern from "../include/aa.h":
    ctypedef struct AaWork:
        pass
    AaWork *aa_init(int, int, int, float, float, float, float, int)
    double aa_apply(double*, const double*, AaWork*)
    int aa_safeguard(double*, double*, AaWork*)
    void aa_reset(AaWork*)
    void aa_finish(AaWork *)

cdef class AndersonAccelerator(object):
    cdef AaWork* _wrk
    cdef int _dim
    def __cinit__(self, dim, mem, type1=False, regularization=1e-9,
                  relaxation=1.0, safeguard_factor=2.0, max_weight_norm=1e4, 
                  verbosity=0):
        self._wrk = aa_init(dim, mem, type1, regularization, relaxation, 
                            safeguard_factor, max_weight_norm, verbosity)
        self._dim = dim

    def _validate(self, f, x):
        f = np.squeeze(f)
        x = np.squeeze(x)
        if (f.shape != (self._dim,) or x.shape != (self._dim,)):
          raise ValueError("Incorrect input dimension")

        if not f.flags['C_CONTIGUOUS']:
            # Makes a contiguous copy of the numpy array.
            f = np.ascontiguousarray(f)
        if not x.flags['C_CONTIGUOUS']:
            # Makes a contiguous copy of the numpy array.
            x = np.ascontiguousarray(x)
        
        return f, x
 
    def apply(self, f, x):
        f, x = self._validate(f, x) 
        cdef double[::1] f_memview = f
        cdef double[::1] x_memview = x
        return aa_apply(&f_memview[0], &x_memview[0], self._wrk)

    def safeguard(self, f_new, x_new):
        f_new , x_new = self._validate(f_new, x_new) 
        cdef double[::1] f_memview = f_new
        cdef double[::1] x_memview = x_new
        return aa_safeguard(&f_memview[0], &x_memview[0], self._wrk)

    def reset(self):
        aa_reset(self._wrk)

    def __dealloc__(self):
        aa_finish(self._wrk)

