/* Gradient descent (GD) on convex quadratic */
#include "aa.h"
#include "aa_blas.h"
#include <getopt.h>
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

/* uniform random number in [-1,1] */
static aa_float rand_float(void) {
  return 2 * (((aa_float)rand()) / RAND_MAX) - 1;
}

static void print_usage(const char *prog) {
  fprintf(stderr,
    "Usage: %s [OPTIONS]\n"
    "\n"
    "Gradient descent with Anderson Acceleration on a random convex quadratic.\n"
    "\n"
    "Options:\n"
    "  -m, --mem=N                   AA memory / lookback (default %d)\n"
    "      --type1                   use type-I AA (default)\n"
    "      --type2                   use type-II AA\n"
    "  -n, --dim=N                   problem dimension (default %d)\n"
    "  -s, --step=F                  gradient step size (default %g)\n"
    "      --seed=N                  PRNG seed (default %d)\n"
    "      --iters=N                 iteration count (default %d)\n"
    "      --regularization=F        AA regularization (default %g)\n"
    "      --relaxation=F            AA mixing in [0, 2] (default %g)\n"
    "      --safeguard-tolerance=F   AA safeguard factor (default %g)\n"
    "      --max-aa-norm=F           AA weight-norm cap (default %g)\n"
    "  -h, --help                    show this message and exit\n",
    prog, MEM, DIM, (double)STEPSIZE, SEED, ITERS,
    (double)REGULARIZATION, (double)RELAXATION,
    (double)SAFEGUARD_TOLERANCE, (double)MAX_AA_NORM);
}

int main(int argc, char **argv) {
  aa_int type1 = TYPE1, n = DIM, iters = ITERS, memory = MEM, seed = SEED;
  aa_int i, one = 1;
  aa_int verbosity = VERBOSITY;
  aa_float step_size = STEPSIZE;
  aa_float regularization = REGULARIZATION;
  aa_float relaxation = RELAXATION;
  aa_float safeguard_tolerance = SAFEGUARD_TOLERANCE;
  aa_float max_aa_norm = MAX_AA_NORM;
  aa_float neg_step_size;
  aa_float err = 0;
  aa_float *x, *xprev, *Qhalf, *Q, zerof = 0.0, onef = 1.0;
  _timer aa_timer;
  aa_float aa_time = 0;

  static struct option long_opts[] = {
    {"mem",                 required_argument, 0, 'm'},
    {"type1",               no_argument,       0, '1'},
    {"type2",               no_argument,       0, '2'},
    {"dim",                 required_argument, 0, 'n'},
    {"step",                required_argument, 0, 's'},
    {"seed",                required_argument, 0,  1 },
    {"iters",               required_argument, 0,  2 },
    {"regularization",      required_argument, 0,  3 },
    {"relaxation",          required_argument, 0,  4 },
    {"safeguard-tolerance", required_argument, 0,  5 },
    {"max-aa-norm",         required_argument, 0,  6 },
    {"help",                no_argument,       0, 'h'},
    {0, 0, 0, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "m:n:s:h", long_opts, NULL)) != -1) {
    switch (opt) {
    case 'm': memory = atoi(optarg); break;
    case '1': type1 = 1; break;
    case '2': type1 = 0; break;
    case 'n': n = atoi(optarg); break;
    case 's': step_size = atof(optarg); break;
    case  1 : seed = atoi(optarg); break;
    case  2 : iters = atoi(optarg); break;
    case  3 : regularization = atof(optarg); break;
    case  4 : relaxation = atof(optarg); break;
    case  5 : safeguard_tolerance = atof(optarg); break;
    case  6 : max_aa_norm = atof(optarg); break;
    case 'h': print_usage(argv[0]); return 0;
    default:  print_usage(argv[0]); return 2;
    }
  }
  neg_step_size = -step_size;

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

  BLAS(gemm)
  ("Trans", "No", &n, &n, &n, &onef, Qhalf, &n, Qhalf, &n, &zerof, Q, &n);

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
    ("No", &n, &n, &neg_step_size, Q, &n, xprev, &one, &onef, x, &one);

    _tic(&aa_timer);
    aa_safeguard(x, xprev, a);
    aa_time += _tocq(&aa_timer);

    err = BLAS(nrm2)(&n, x, &one);
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
