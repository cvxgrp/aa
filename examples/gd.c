/* Gradient descent (GD) on convex quadratic */
#include "aa.h"
#include "aa_blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* default parameters */
#define SEED (1)
#define TYPE1 (1)
#define DIM (100)
#define MEM (10)
#define ITERS (1000)
#define STEPSIZE (0.01)
#define PRINT (100)

/* uniform random number in [-1,1] */
static aa_float rand_float(void) {
  return 2 * (((aa_float)rand()) / RAND_MAX) - 1;
}

/*
 * out/gd memory dimension step_size type1 seed iters
 *
 */
int main(int argc, char **argv) {
  aa_int type1 = TYPE1, n = DIM, iters = ITERS, memory = MEM, seed = SEED;
  aa_int i, one = 1;
  aa_float err, neg_step_size = -STEPSIZE;
  aa_float *x, *xprev, *Qhalf, *Q, zerof = 0.0, onef = 1.0;

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
    printf("Usage: 'out/gd memory dimension step_size type1 seed iters'\n");
    break;
  }

  x = malloc(sizeof(aa_float) * n);
  xprev = malloc(sizeof(aa_float) * n);
  Qhalf = malloc(sizeof(aa_float) * n * n);
  Q = malloc(sizeof(aa_float) * n * n);

  srand(seed);

  /* generate random data */
  for (i = 0; i < n; i++) {
    x[i] = rand_float();
  }
  for (i = 0; i < n * n; i++) {
    Qhalf[i] = rand_float();
  }

  BLAS(gemm)
  ("Trans", "No", &n, &n, &n, &onef, Qhalf, &n, Qhalf, &n, &zerof, Q, &n);

  /* add some regularization */
  for (i = 0; i < n; i++) {
    Q[i + i * n] += 1.0;
  }

  AaWork *a = aa_init(n, memory, type1);
  for (i = 0; i < iters; i++) {
    memcpy(xprev, x, sizeof(aa_float) * n);
    /* x = x - step_size * Q * xprev */
    BLAS(gemv)
    ("No", &n, &n, &neg_step_size, Q, &n, xprev, &one, &onef, x, &one);

    aa_apply(x, xprev, a);

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
  return 0;
}
