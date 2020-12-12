#ifndef AA_H_GUARD
#define AA_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double aa_float;
typedef int aa_int;

typedef struct ACCEL_WORK AaWork;

/* Initialize Anderson Acceleration, allocates memory.
 *
 * Args:
 *  dim: the dimension of the variable for aa
 *  mem: the memory (number of past iterations used) for aa
 *  type1: bool, if True use type 1 aa, otherwise use type 2
 *  regularization: float, regularization param, type-I and type-II different
 *       for type-I: 1e-8 works well, type-II: more stable can use 1e-10 often
 *  verbosity: if greater than 0 prints out various info
 *
 * Reurns:
 *  Pointer to aa workspace
 */
AaWork *aa_init(aa_int dim, aa_int mem, aa_int type1, aa_float regaularization,
                aa_int verbosity);

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

/* Reset Anderson Acceleration.
 *
 * Resets AA, uses original parameters, reuses original memory allocations.
 *
 * Args:
 *  a: aa workspace from aa_init.
 */
void aa_reset(AaWork *a);

#define MAX_AA_NRM (1e3)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#ifdef __cplusplus
}
#endif
#endif
