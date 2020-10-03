AA
====

AA (`Anderson Acceleration`)

C (with python interface) implementation of the Anderson Acceleration algorithm as described in our paper [Globally Convergent Type-I Anderson Acceleration for Non-Smooth Fixed-Point Iterations](https://web.stanford.edu/~boyd/papers/nonexp_global_aa1.html)

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
aa_wrk = aa.AndersonAccelerator(dim, mem, type1, eta)
```
where:
* `dim` is the integer problem dimension.
* `mem` is the integer amount of memory (or lookback) you want the algorithm to use, around 10 is a good number for this. 
* `type1` is a boolean, if `True` uses type-1 AA, otherwise uses type-2 AA.
* `eta`: float, regularization param, type-I: 1e-8 works well, type-II: more stable can use 1e-10 often

To use the accelerator:
```python
aa_wrk.apply(x, x_prev)
```
where:
* `x` is the numpy array consisting of the current iterate and it will be overwritten with the accelerated iterate.
* `x_prev` is the numpy array consisting of the previous iterate (the input to the update function).


C
----

At the command prompt type `make` to compile the library and the example. The
example can be run by `out/gd`.

The C API is as follows:

```C
/* Initialize Anderson Acceleration, allocates memory.
 *
 * Args:
 *  dim: the dimension of the variable for aa
 *  mem: the memory (number of past iterations used) for aa
 *  type1: bool, if True use type 1 aa, otherwise use type 2
 *  eta: float, regularization param, type-I and type-II different
 *       type-I: 1e-8 works well, type-II: more stable can use 1e-10 often
 *
 * Reurns:
 *  Pointer to aa workspace
 */
AaWork *aa_init(aa_int dim, aa_int mem, aa_int type1, aa_float eta);

/* Apply Anderson Acceleration.
 *
 * Args:
 *  f: output of map at current iteration, overwritten with aa output at end.
 *  x: input to map at current iteration
 *  a: aa workspace from aa_init
 *
 * Returns:
 *  int, a value of 0 is success, <0 is failure at which point f is unchanged
 */
aa_int aa_apply(aa_float *f, const aa_float *x, AaWork *a);

/* Finish Anderson Acceleration, clears memory.
 *
 * Args:
 *  a: aa workspace from aa_init.
 */
void aa_finish(AaWork *a);
```

