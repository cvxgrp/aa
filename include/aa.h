#ifndef AA_H_GUARD
#define AA_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "aa_blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NRM (10.0)

#ifndef NULL
#define NULL ((void *)0)
#endif

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define aa_float double
#define aa_int int
#define blas_int int

typedef struct ACCEL_WORK AaWork;

AaWork *aa_init(aa_int l, aa_int aa_mem, aa_int type1);
aa_int aa_apply(const aa_float *x, const aa_float *fx, aa_float *sol,
                AaWork *a);
void aa_finish(AaWork *a);

#ifdef __cplusplus
}
#endif
#endif
