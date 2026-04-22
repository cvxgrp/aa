import math

import numpy as np

cdef extern from "../src/aa.c":
    pass

cdef extern from "../include/aa.h":
    ctypedef struct AaWork:
        pass
    ctypedef struct AaStats:
        int iter
        int n_accept
        int n_reject_lapack
        int n_reject_rank0
        int n_reject_nonfinite
        int n_reject_weight_cap
        int n_safeguard_reject
        int last_rank
        double last_aa_norm
        double last_regularization
    AaWork *aa_init(int, int, int, int, double, double, double, double, int, int)
    double aa_apply(double*, const double*, AaWork*)
    int aa_safeguard(double*, double*, AaWork*)
    void aa_reset(AaWork*)
    void aa_finish(AaWork *)
    AaStats aa_get_stats(const AaWork*)

cdef class AndersonAccelerator(object):
    cdef AaWork* _wrk
    cdef int _dim

    def __cinit__(self, dim, mem, *, min_len=None, type1=False,
                  regularization=1e-12, relaxation=1.0, safeguard_factor=1.0,
                  max_weight_norm=1e6, ir_max_steps=5, verbosity=0):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if mem < 0:
            raise ValueError("mem must be non-negative")
        # regularization accepts any finite value:
        #   > 0  → scaled by ||A||_F ||Y||_F
        #   < 0  → pinned absolute |regularization|
        #   = 0  → off
        if not math.isfinite(regularization):
            raise ValueError("regularization must be finite")
        if relaxation < 0 or relaxation > 2:
            raise ValueError("relaxation must be in [0, 2]")
        if safeguard_factor < 0:
            raise ValueError("safeguard_factor must be non-negative")
        if max_weight_norm <= 0:
            raise ValueError("max_weight_norm must be positive")
        if ir_max_steps < 0:
            raise ValueError("ir_max_steps must be non-negative")
        # min_len: minimum # of residual pairs before AA starts extrapolating.
        # Default = min(mem, dim), preserving the historical "wait until the
        # memory is full" behavior. aa_init will clamp down when
        # min_len > min(mem, dim), matching how it already clamps `mem`.
        if min_len is None:
            min_len = min(mem, dim)
        if mem > 0 and min_len < 1:
            raise ValueError("min_len must be >= 1 when mem > 0")
        self._wrk = aa_init(dim, mem, min_len, type1, regularization,
                            relaxation, safeguard_factor, max_weight_norm,
                            ir_max_steps, verbosity)
        if self._wrk is NULL:
            raise MemoryError("aa_init failed")
        self._dim = dim

    def _normalize_array(self, arr, name):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if arr.dtype != np.float64:
            raise TypeError("arrays must be float64")
        if arr.shape == (self._dim,):
            view = arr
        elif arr.shape == (self._dim, 1) or arr.shape == (1, self._dim):
            view = arr.reshape(self._dim)
        else:
            raise ValueError("Incorrect input dimension")
        # aa_apply / aa_safeguard write through these pointers, so we must
        # not silently hand the C code a copy. Reject non-contiguous input
        # rather than copying and dropping the writes on the floor.
        if not view.flags['C_CONTIGUOUS'] or not view.flags['WRITEABLE']:
            raise ValueError(f"{name} must be C-contiguous and writeable")
        return view

    def _validate(self, f, x):
        return self._normalize_array(f, "first array"), self._normalize_array(x, "second array")

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

    @property
    def stats(self):
        """Lifetime diagnostic counters. Not cleared by reset().

        Returns a dict with:
          iter                 : internal iteration counter
          n_accept             : # of AA steps accepted by apply()
          n_reject_lapack      : # of solves rejected due to LAPACK geqp3 failure
          n_reject_rank0       : # of solves where pivoted QR truncated to rank 0
          n_reject_nonfinite   : # of solves with non-finite ||γ||₂
          n_reject_weight_cap  : # of solves with ||γ||₂ >= max_weight_norm
          n_safeguard_reject   : # of safeguard rollbacks
          last_rank            : numerical rank of most recent LS solve
          last_aa_norm         : ||γ||₂ of most recent solve (NaN if none yet
                                 or solve failed)
          last_regularization  : r used in most recent solve (0 if none yet)
        """
        cdef AaStats s = aa_get_stats(self._wrk)
        return {
            "iter": int(s.iter),
            "n_accept": int(s.n_accept),
            "n_reject_lapack": int(s.n_reject_lapack),
            "n_reject_rank0": int(s.n_reject_rank0),
            "n_reject_nonfinite": int(s.n_reject_nonfinite),
            "n_reject_weight_cap": int(s.n_reject_weight_cap),
            "n_safeguard_reject": int(s.n_safeguard_reject),
            "last_rank": int(s.last_rank),
            "last_aa_norm": float(s.last_aa_norm),
            "last_regularization": float(s.last_regularization),
        }

    def __dealloc__(self):
        aa_finish(self._wrk)
