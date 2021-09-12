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
 *  relaxation: float \in [0,2], mixing parameter (1.0 is vanilla AA)
 *  safeguard_tolerance: float, factor that controls safeguarding checks, larger
 *        is more aggressive but less stable
 *  max_aa_norm: float, maximum norm of aa weights
 *  verbosity: if greater than 0 prints out various info
 *
 * Reurns:
 *  Pointer to aa workspace
 */
AaWork *aa_init(aa_int dim, aa_int mem, aa_int type1, aa_float regularization,
                aa_float relaxation, aa_float safeguard_tolerance,
                aa_float max_aa_norm, aa_int verbosity);

/* Apply Anderson Acceleration.
 *
 * Args:
 *  f: output of map at current iteration, overwritten with aa output at end.
 *  x: input to map at current iteration
 *  a: workspace from aa_init
 *
 * Returns:
 *  (float) (+ or -) norm of AA weights vector:
 *    if positive then update was accepted and f contains new point
 *    if negative then update was rejected and f is unchanged
 */
aa_float aa_apply(aa_float *f, const aa_float *x, AaWork *a);

/* Apply safeguarding.
 *
 * This step is optional but can improve stability. The pattern is as follows:
 *
 *    aa_apply(x, x_prev, a)
 *    x_prev = x
 *    x = F(x)
 *    aa_safeguard(x, x_prev, a)
 *
 * Args:
 *  f_new: output of map after AA step
 *  x_new: input to map after AA step
 *  a: workspace from aa_init
 *
 * Returns:
 *  (int) 0 if AA step is accepted, otherwise -1
 *        if AA step is rejected then overwite f_new and x_new with previous
 *          values
 */
aa_int aa_safeguard(aa_float *f_new, aa_float *x_new, AaWork *a);

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

#ifdef __cplusplus
}
#endif
#endif
