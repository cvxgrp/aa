AA
====

AA (`Anderson Acceleration`)

Code based on our paper: https://web.stanford.edu/~boyd/papers/nonexp_global_aa1.html

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
 *
 * Reurns:
 *  Pointer to aa workspace
 */
AaWork *aa_init(aa_int dim, aa_int mem, aa_int type1);

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

