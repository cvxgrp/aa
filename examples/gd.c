/* Gradient descent (GD) on convex quadratic */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "aa.h"

#define SEED (0)
#define DIM (100)
#define MEM (10)
#define ITERS (1000000)
#define STEPSIZE (0.01)

aa_float BLAS(nrm2)(aa_int *n, aa_float *x, aa_int *incx);
void BLAS(axpy)(aa_int *n, aa_float *a, const aa_float *x, aa_int *incx,
                aa_float *y, aa_int *incy);
void BLAS(gemv)(const char *trans, const aa_int *m, const aa_int *n,
                const aa_float *alpha, const aa_float *a, const aa_int *lda,
                const aa_float *x, const aa_int *incx, const aa_float *beta,
                aa_float *y, const aa_int *incy);
void BLAS(gemm)(const char *transa, const char *transb, aa_int *m, aa_int *n,
                aa_int *k, aa_float *alpha, aa_float *a, aa_int *lda,
                aa_float *b, aa_int *ldb, aa_float *beta, aa_float *c,
                aa_int *ldc);

/* uniform random number in [-1,1] */
static aa_float rand_float(void) {
  return 2 * (((aa_float)rand()) / RAND_MAX) - 1;
}

int main(int argc, char **argv) {
  aa_int i, n = DIM, one = 1;
  aa_float zerof = 0.0, onef = 1.0;
  aa_float err, neg_step_size = -STEPSIZE;
  aa_float *x = malloc(sizeof(aa_float) * DIM);
  aa_float *Qhalf = malloc(sizeof(aa_float) * DIM * DIM);
  aa_float *Q = malloc(sizeof(aa_float) * DIM * DIM);
  aa_float *xprev = malloc(sizeof(aa_float) * DIM);
  aa_float *xt = malloc(sizeof(aa_float) * DIM);

  srand(SEED);

  for (i=0; i < DIM; i++) {
    x[i] = rand_float();
  }
  for (i=0; i < DIM * DIM; i++) {
    Qhalf[i] = rand_float();
  }

  BLAS(gemm)("Trans", "No", &n, &n, &n, &onef, Qhalf, &n, Qhalf, &n, &zerof, Q, &n);

  AaWork * a = aa_init(n, MEM);
  for (i=0; i < ITERS; i++) {
    memcpy(xprev, x, sizeof(aa_float) * n);
    /* x = x + neg_step_size * Q * xprev */
    BLAS(gemv)("No", &n, &n, &neg_step_size, Q, &n, xprev, &one, &onef, x, &one);

    aa_apply(xprev, x, xt, a);
    memcpy(x, xt, sizeof(aa_float) * n);

    err = BLAS(nrm2)(&n, x, &one);
    if (i % 1000 == 0) {
      printf("Iter: %i, Err %.4e\n", i, err);
    }
  }
  aa_finish(a);
  return 0;
}

