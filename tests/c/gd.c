/* Gradient descent (GD) on convex quadratic */
#include "aa.h"
#include "aa_blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* default parameters */
#define SEED (1234)
#define TYPE1 (1)
#define DIM (1000)
#define MEM (5)
#define REGULARIZATION (0)
#define SAFEGUARD_TOLERANCE (2.0)
#define MAX_AA_NORM (1e10)
#define RELAXATION (1.0)
#define ITERS (30000)
#define STEPSIZE (0.001)
#define PRINT_INTERVAL (500)
#define VERBOSITY (1)


#ifdef _WIN32
#include <windows.h>
typedef struct _timer {
  LARGE_INTEGER tic;
  LARGE_INTEGER toc;
} _timer;

void _tic(_timer *t) {
  QueryPerformanceCounter(&t->tic);
}

aa_float _tocq(_timer *t) {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&t->toc);
  return (aa_float)(t->toc.QuadPart - t->tic.QuadPart) / (aa_float)freq.QuadPart * 1e3;
}
#else
/* duplicate these with underscore prefix */
typedef struct _timer {
  struct timespec tic;
  struct timespec toc;
} _timer;

void _tic(_timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->tic);
}

aa_float _tocq(_timer *t) {
  struct timespec temp;

  clock_gettime(CLOCK_MONOTONIC, &t->toc);

  if ((t->toc.tv_nsec - t->tic.tv_nsec) < 0) {
    temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec - 1;
    temp.tv_nsec = 1e9 + t->toc.tv_nsec - t->tic.tv_nsec;
  } else {
    temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec;
    temp.tv_nsec = t->toc.tv_nsec - t->tic.tv_nsec;
  }
  return (aa_float)temp.tv_sec * 1e3 + (aa_float)temp.tv_nsec / 1e6;
}
#endif

/* uniform random number in [-1,1] */
static aa_float rand_float(void) {
  return 2 * (((aa_float)rand()) / RAND_MAX) - 1;
}

/*
 * out/gd memory dimension step_size type1 seed iters regularization
 *
 */
int main(int argc, char **argv) {
  aa_int type1 = TYPE1, n = DIM, iters = ITERS, memory = MEM, seed = SEED;
  aa_int i;
  aa_int verbosity = VERBOSITY;
  blas_int bn, bone = 1;
  aa_float neg_step_size = -STEPSIZE;
  aa_float regularization = REGULARIZATION;
  aa_float relaxation = RELAXATION;
  aa_float safeguard_tolerance = SAFEGUARD_TOLERANCE;
  aa_float max_aa_norm = MAX_AA_NORM;
  aa_float err = 0;
  aa_float *x, *xprev, *Qhalf, *Q, zerof = 0.0, onef = 1.0;
  _timer aa_timer;
  aa_float aa_time = 0;

  printf("Usage: 'out/gd memory type1 dimension step_size seed iters "
         "regularization relaxation safeguard_tolerance max_aa_norm'\n");

  switch (argc - 1) {
  case 10:
    max_aa_norm = atof(argv[10]);
  case 9:
    safeguard_tolerance = atof(argv[9]);
  case 8:
    relaxation = atof(argv[8]);
  case 7:
    regularization = atof(argv[7]);
  case 6:
    iters = atoi(argv[6]);
  case 5:
    seed = atoi(argv[5]);
  case 4:
    neg_step_size = -atof(argv[4]);
  case 3:
    n = atoi(argv[3]);
  case 2:
    type1 = atoi(argv[2]);
  case 1:
    memory = atoi(argv[1]);
    break;
  default:
    printf("Running default parameters.\n");
  }

  x = (aa_float *)malloc(sizeof(aa_float) * n);
  xprev = (aa_float *)malloc(sizeof(aa_float) * n);
  Qhalf = (aa_float *)malloc(sizeof(aa_float) * n * n);
  Q = (aa_float *)malloc(sizeof(aa_float) * n * n);

  srand(seed);

  /* generate random data */
  for (i = 0; i < n; i++) {
    x[i] = rand_float();
  }
  for (i = 0; i < n * n; i++) {
    Qhalf[i] = rand_float();
  }

  bn = (blas_int)n;
  BLAS(gemm)
  ("Trans", "No", &bn, &bn, &bn, &onef, Qhalf, &bn, Qhalf, &bn, &zerof, Q, &bn);

  /* add small amount regularization */
  for (i = 0; i < n; i++) {
    Q[i + i * n] += 1e-6;
  }

  AaWork *a = aa_init(n, memory, type1, regularization, relaxation,
                      safeguard_tolerance, max_aa_norm, verbosity);
  for (i = 0; i < iters; i++) {
    if (i > 0) {
      _tic(&aa_timer);
      aa_apply(x, xprev, a);
      aa_time += _tocq(&aa_timer);
    }

    memcpy(xprev, x, sizeof(aa_float) * n);
    /* x = x - step_size * Q * xprev */
    BLAS(gemv)
    ("No", &bn, &bn, &neg_step_size, Q, &bn, xprev, &bone, &onef, x, &bone);

    _tic(&aa_timer);
    aa_safeguard(x, xprev, a);
    aa_time += _tocq(&aa_timer);

    err = BLAS(nrm2)(&bn, x, &bone);
    if (i % PRINT_INTERVAL == 0) {
      printf("Iter: %i, Err %.4e\n", i, err);
    }
  }
  printf("Iter: %i, Err %.4e\n", i, err);
  printf("AA time: %.4f seconds\n", aa_time / 1e3);
  aa_finish(a);
  free(Q);
  free(Qhalf);
  free(x);
  free(xprev);
  return 0;
}
