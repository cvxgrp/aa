/* Gradient descent (GD) on convex quadratic */
#include "aa.h"
#include "aa_blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* default parameters */
#define SEED (1)
#define TYPE1 (0)
#define DIM (100)
#define MEM (10)
#define ETA (1e-9)
#define ITERS (1000)
#define STEPSIZE (0.01)
#define PRINT (100)
#define INTERVAL (10)

/* uniform random number in [-1,1] */
static aa_float rand_float(void) {
  return 2 * (((aa_float)rand()) / RAND_MAX) - 1;
}

/*
 * out/gd memory dimension step_size type1 seed iters eta
 *
 */
int main(int argc, char **argv) {
  aa_int type1 = TYPE1, n = DIM, iters = ITERS, memory = MEM, seed = SEED;
  aa_int i, one = 1, interval = INTERVAL;
  aa_float err, neg_step_size = -STEPSIZE, eta = ETA;
  aa_float *x, *xprev, *Qhalf, *Q, zerof = 0.0, onef = 1.0;
  struct timespec tic, toc, temp;

  switch (argc) {
  case 9:
    interval = atoi(argv[8]);
  case 8:
    eta = atof(argv[7]);
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
    break;
  }
  printf("Usage: 'out/gd memory dimension step_size type1 seed iters eta "
         "interval'\n");

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

  clock_gettime(CLOCK_MONOTONIC, &tic);
  AaWork *a = aa_init(n, memory, type1, interval, eta);
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

  clock_gettime(CLOCK_MONOTONIC, &toc);
  if ((toc.tv_nsec - tic.tv_nsec) < 0) {
    temp.tv_sec = toc.tv_sec - tic.tv_sec - 1;
    temp.tv_nsec = 1e9 + toc.tv_nsec - tic.tv_nsec;
  } else {
    temp.tv_sec = toc.tv_sec - tic.tv_sec;
    temp.tv_nsec = toc.tv_nsec - tic.tv_nsec;
  }
  printf("gd run-time: %8.4f ms.\n", temp.tv_sec * 1e3 + temp.tv_nsec / 1e6);

  free(Q);
  free(Qhalf);
  free(x);
  free(xprev);
  
  return 0;
}
