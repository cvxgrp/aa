/* Gradient descent (GD) on convex quadratic */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "aa.h"

#define SEED (1)
#define TYPE1 (1)
#define DIM (100)
#define MEM (10)
#define ITERS (1000)
#define STEPSIZE (0.01)
#define PRINT (100)


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

/*
 * out/gd memory dimension step_size seed iters
 *
 */
int main(int argc, char **argv) {
  aa_int i, type1 = TYPE1, n = DIM, iters = ITERS, memory = MEM, seed = SEED, one = 1;
  aa_float err, neg_step_size = -STEPSIZE;
  aa_float *x, *xprev, *xt, *Qhalf, *Q, zerof = 0.0, onef = 1.0;

  switch (argc) {
    case 7:
      iters = atoi(argv[6]);
    case 6:
      seed = atoi(argv[5]);
    case 5:
      type1 = atoi(argv[4]);
    case 4:
      neg_step_size = -atof(argv[3]);
    case 3:
      n = atoi(argv[2]);
    case 2:
      memory = atoi(argv[1]);
      break;
    default:
      printf("Running default parameters.\n");
      printf("Usage: 'out/gd memory dimension step_size seed iters'\n");
      break;
  }

  x = malloc(sizeof(aa_float) * n);
  xprev = malloc(sizeof(aa_float) * n);
  xt = malloc(sizeof(aa_float) * n);
  Qhalf = malloc(sizeof(aa_float) * n * n);
  Q = malloc(sizeof(aa_float) * n * n);

  srand(seed);

  for (i=0; i < n; i++) {
    x[i] = rand_float();
  }
  for (i=0; i < n * n; i++) {
    Qhalf[i] = rand_float();
  }

  BLAS(gemm)("Trans", "No", &n, &n, &n, &onef, Qhalf, &n, Qhalf, &n, &zerof, Q, &n);

  for (i=0; i < n; i++) {
    Q[i + i * n] += 1.0;
  }

  AaWork * a = aa_init(n, memory, type1);
  for (i=0; i < iters; i++) {
    memcpy(xprev, x, sizeof(aa_float) * n);
    /* x = x - step_size * Q * xprev */
    BLAS(gemv)("No", &n, &n, &neg_step_size, Q, &n, xprev, &one, &onef, x, &one);

    aa_apply(xprev, x, xt, a);
    memcpy(x, xt, sizeof(aa_float) * n);

    err = BLAS(nrm2)(&n, x, &one);
    if (i % PRINT == 0) {
      printf("Iter: %i, Err %.4e\n", i, err);
    }
  }
  aa_finish(a);
  free(Q);
  free(Qhalf);
  free(x);
  free(xprev);
  free(xt);
  return 0;
}

