AA
===

[![Build Status](https://github.com/cvxgrp/aa/actions/workflows/build.yml/badge.svg)](https://github.com/cvxgrp/aa/actions/workflows/build.yml)

AA (`Anderson Acceleration`)

C (with python interface) implementation of the Anderson Acceleration algorithm as described in our paper [Globally Convergent Type-I Anderson Acceleration for Non-Smooth Fixed-Point Iterations](https://web.stanford.edu/~boyd/papers/nonexp_global_aa1.html)

NOTE: This implementation is a simple proof-of-concept and does not include all
the necessary stabilizations required to guarantee convergence. However, it
works well in many cases.

MATLAB code (and the experiments presented in the paper) available [here](https://github.com/cvxgrp/nonexp_global_aa1/): 

----

Python
----

To install the package use:
```bash
cd python
python setup.py install
```
To test, run in the same directory:
```bash
python example.py
```

The Python API is as follows. To initialize the accelerator:
```python
import aa
aa_wrk = aa.AndersonAccelerator(dim, mem, type1=False, regularization=1e-12,
                                relaxation=1.0, safeguard_factor=1.0,
                                max_weight_norm=1e6, verbosity=0)
```
where:
* `dim`: integer problem dimension.
* `mem`: integer amount of memory (or lookback) to use; around 10 is a good starting point.
* `type1`: bool, if `True` uses type-I AA, otherwise type-II.
* `regularization`: float, regularization param (multiplied internally by `||M||_F`); type-I: 1e-8 works well, type-II can use 1e-10 / 1e-12.
* `relaxation`: float in `[0, 2]`, mixing parameter (1.0 is vanilla AA).
* `safeguard_factor`: float, tolerance used by `safeguard`; larger is more aggressive but less stable.
* `max_weight_norm`: float, reject AA steps whose weight vector norm exceeds this.
* `verbosity`: int, if greater than 0 prints diagnostic info.

To use the accelerator:
```python
aa_wrk.apply(x, x_prev)                  # in-place accelerate x
aa_wrk.safeguard(x_after_map, x_prev)    # optional stability check
aa_wrk.reset()                           # discard history, keep workspace
```
* `apply(x, x_prev)`: `x` is the current iterate (output of the map applied to `x_prev`), overwritten in place with the accelerated iterate; `x_prev` is the previous iterate.
* `safeguard(f_new, x_new)`: optional safeguard after applying the map to the accelerated point. If rejected, `f_new` and `x_new` are restored to their pre-AA values.
* Arrays passed to `apply`/`safeguard` must be C-contiguous `float64`.


C
----

At the command prompt type `make` to compile the library and the example. The
example can be run by `out/gd`.

The C API (see `include/aa.h` for full docstrings) is:

```C
/* Initialize an AA workspace; returns NULL on invalid params or alloc failure. */
AaWork *aa_init(aa_int dim, aa_int mem, aa_int type1, aa_float regularization,
                aa_float relaxation, aa_float safeguard_factor,
                aa_float max_weight_norm, aa_int verbosity);

/* Apply AA. f is overwritten with the accelerated iterate; x is the previous
 * iterate (input to the map). Returns the (signed) norm of the AA weights:
 * positive => accepted, negative => rejected and f is unchanged. */
aa_float aa_apply(aa_float *f, const aa_float *x, AaWork *a);

/* Optional safeguard: if the AA step made the residual grow too much, revert
 * f_new and x_new to the pre-AA values. Returns 0 on accept, -1 on reject. */
aa_int aa_safeguard(aa_float *f_new, aa_float *x_new, AaWork *a);

/* Discard history; keep workspace allocations. Useful after a divergent step. */
void aa_reset(AaWork *a);

/* Free the workspace. */
void aa_finish(AaWork *a);
```

`AaWork` is not thread-safe: one workspace per thread.

