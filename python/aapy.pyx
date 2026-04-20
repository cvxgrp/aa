import numpy as np

cdef extern from "../src/aa.c":
    pass

cdef extern from "../include/aa.h":
    ctypedef struct AaWork:
        pass
    AaWork *aa_init(int, int, int, double, double, double, double, int)
    double aa_apply(double*, const double*, AaWork*)
    int aa_safeguard(double*, double*, AaWork*)
    void aa_reset(AaWork*)
    void aa_finish(AaWork *)

cdef class AndersonAccelerator(object):
    cdef AaWork* _wrk
    cdef int _dim

    def __cinit__(self, dim, mem, type1=False, regularization=1e-12,
                  relaxation=1.0, safeguard_factor=1.0, max_weight_norm=1e6,
                  verbosity=0):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if mem < 0:
            raise ValueError("mem must be non-negative")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        if relaxation < 0 or relaxation > 2:
            raise ValueError("relaxation must be in [0, 2]")
        if safeguard_factor < 0:
            raise ValueError("safeguard_factor must be non-negative")
        if max_weight_norm <= 0:
            raise ValueError("max_weight_norm must be positive")
        self._wrk = aa_init(dim, mem, type1, regularization, relaxation,
                            safeguard_factor, max_weight_norm, verbosity)
        if self._wrk is NULL:
            raise MemoryError("aa_init failed")
        self._dim = dim

    def _validate(self, f, x):
        # atleast_1d undoes np.squeeze's collapse of shape-(1,) to shape-() —
        # without it, dim=1 users cannot pass a 1-D array through validation.
        f = np.atleast_1d(np.squeeze(f))
        x = np.atleast_1d(np.squeeze(x))
        if f.shape != (self._dim,) or x.shape != (self._dim,):
            raise ValueError("Incorrect input dimension")
        # aa_apply / aa_safeguard write through these pointers, so we must
        # not silently hand the C code a copy. Reject non-contiguous input
        # rather than copying and dropping the writes on the floor.
        if not f.flags['C_CONTIGUOUS'] or not f.flags['WRITEABLE']:
            raise ValueError("first array must be C-contiguous and writeable")
        if not x.flags['C_CONTIGUOUS'] or not x.flags['WRITEABLE']:
            raise ValueError("second array must be C-contiguous and writeable")
        if f.dtype != np.float64 or x.dtype != np.float64:
            raise TypeError("arrays must be float64")
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
