/* Gradient descent (GD) on convex quadratic */
#include "aa.h"
#include "aa_blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* default parameters */
#define SEED (1234)
#define DIM (1000)
#define MEM (5)
#define SAFEGUARD_TOLERANCE (1.0)
#define ITERS (30000)
#define STEPSIZE (0.001)
#define PRINT_INTERVAL (500)
#define VERBOSITY (1)

/* uniform random number in [-1,1] */
static aa_float rand_float(void) {
  return 2 * (((aa_float)rand()) / RAND_MAX) - 1;
}

/*
 * out/gd memory dimension step_size type1 seed iters regularization
 *
 */
int main(int argc, char **argv) {
  aa_int n = DIM, iters = ITERS, memory = MEM, seed = SEED;
  aa_int i, one = 1;
  aa_int verbosity = VERBOSITY;
  aa_float neg_step_size = -STEPSIZE;
  aa_float safeguard_tolerance = SAFEGUARD_TOLERANCE;
  aa_float err;
  aa_float *x, *xprev, *Qhalf, *Q, zerof = 0.0, onef = 1.0;

  printf("Usage: 'out/gd memory type1 dimension step_size seed iters safeguard_tolerance'\n");

  switch (argc-1) {
  case 6:
    safeguard_tolerance = atof(argv[6]);
  case 5:
    iters = atoi(argv[5]);
  case 4:
    seed = atoi(argv[4]);
  case 3:
    neg_step_size = -atof(argv[3]);
  case 2:
    n = atoi(argv[2]);
  case 1:
    memory = atoi(argv[1]);
    break;
  default:
    printf("Running default parameters.\n");
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

  /* add small amount regularization */
  for (i = 0; i < n; i++) {
    Q[i + i * n] += 1e-6;
  }

  AaWork *a = aa_init(n, memory, safeguard_tolerance, verbosity);
  for (i = 0; i < iters; i++) {
    if (i > 0) {
      aa_apply(x, xprev, a);
    }

    memcpy(xprev, x, sizeof(aa_float) * n);
    /* x = x - step_size * Q * xprev */
    BLAS(gemv)
    ("No", &n, &n, &neg_step_size, Q, &n, xprev, &one, &onef, x, &one);

    aa_safeguard(x, xprev, a);

    err = BLAS(nrm2)(&n, x, &one);
    if (i % PRINT_INTERVAL == 0) {
      printf("Iter: %i, Err %.4e\n", i, err);
    }
  }
  printf("Iter: %i, Err %.4e\n", i, err);
  aa_finish(a);
  free(Q);
  free(Qhalf);
  free(x);
  free(xprev);
  return 0;
}
