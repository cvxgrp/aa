/* Test suite for libaa.
 *
 * Two kinds of tests live here:
 *   1. Integration: gradient descent on a random convex quadratic. These
 *      exercise the full AA pipeline (BLAS calls, type-I/type-II, relaxed
 *      and unrelaxed) and check convergence.
 *   2. Unit / edge-case: targeted tests for aa_init, aa_reset, and the
 *      boundary behaviors of aa_apply / aa_safeguard that are easy to
 *      regress when touching the internals.
 */
#include "aa.h"
#include "aa_blas.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "minunit.h"

/* default parameters for the integration tests */
#define SEED (1234)
#define DIM (100)
#define MEM (5)
#define TYPE1_REGULARIZATION (1e-3)
#define TYPE2_REGULARIZATION (0)
#define SAFEGUARD_TOLERANCE (2.0)
#define MAX_AA_NORM (1e10)
#define ITERS (10000)
#define STEPSIZE (0.01)
#define PRINT_INTERVAL (500)
#define VERBOSITY (1)

int tests_run = 0;

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

static aa_float nrm2_vec(const aa_float *x, aa_int n) {
  aa_float s = 0;
  for (aa_int i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return sqrt(s);
}

/* =============================================================== */
/*  Integration tests: GD on random convex quadratic.              */
/* =============================================================== */

static const char *gd(aa_int type1, aa_float relaxation) {
  aa_int n = DIM, iters = ITERS, memory = MEM, seed = SEED;
  aa_int i;
  aa_int verbosity = VERBOSITY;
  blas_int bn = (blas_int)n, bone = 1;
  aa_float neg_step_size = -STEPSIZE;
  aa_float safeguard_tolerance = SAFEGUARD_TOLERANCE;
  aa_float max_aa_norm = MAX_AA_NORM;
  aa_float err = 0;
  aa_float regularization;
  aa_float *x, *xprev, *Qhalf, *Q, zerof = 0.0, onef = 1.0;
  _timer aa_timer;
  aa_float aa_time = 0;
  x = (aa_float *)malloc(sizeof(aa_float) * n);
  xprev = (aa_float *)malloc(sizeof(aa_float) * n);
  Qhalf = (aa_float *)malloc(sizeof(aa_float) * n * n);
  Q = (aa_float *)malloc(sizeof(aa_float) * n * n);

  srand(seed);

  if (type1) {
    regularization = TYPE1_REGULARIZATION;
  } else {
    regularization = TYPE2_REGULARIZATION;
  }

  /* generate random data */
  for (i = 0; i < n; i++) {
    x[i] = rand_float();
  }
  for (i = 0; i < n * n; i++) {
    Qhalf[i] = rand_float();
  }

  BLAS(gemm)
  ("Trans", "No", &bn, &bn, &bn, &onef, Qhalf, &bn, Qhalf, &bn, &zerof, Q, &bn);

  /* add small amount regularization */
  for (i = 0; i < n; i++) {
    Q[i + i * n] += 1e-2;
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

  mu_assert_less("Failed to produce small error", err, 1e-6);

  return 0;
}

static const char *gd_type1_relax1(void)  { return gd(1, 1.0);  }
static const char *gd_type1_relaxl1(void) { return gd(1, 0.98); }
static const char *gd_type2_relax1(void)  { return gd(0, 1.0);  }
static const char *gd_type2_relaxl1(void) { return gd(0, 0.98); }

/* =============================================================== */
/*  Unit / edge-case tests.                                        */
/* =============================================================== */

/* Run diagonal-Q gradient descent (optionally with AA) and return
 * ||x|| after `iters` iterations. Qdiag must have strictly positive
 * entries; step is a fixed scalar. Pass mem=0 to disable AA. */
static aa_float diag_gd(const aa_float *Qdiag, aa_int n, aa_float step,
                        aa_int mem, aa_int type1, aa_float relaxation,
                        aa_int iters, unsigned seed) {
  aa_float *x = (aa_float *)calloc(n, sizeof(aa_float));
  aa_float *xprev = (aa_float *)calloc(n, sizeof(aa_float));
  srand(seed);
  for (aa_int i = 0; i < n; i++) {
    x[i] = rand_float();
  }
  AaWork *a = aa_init(n, mem, type1, /*reg=*/1e-10, relaxation,
                      /*safeguard=*/2.0, /*max_w=*/1e10, /*verbosity=*/0);
  for (aa_int i = 0; i < iters; i++) {
    if (i > 0) {
      aa_apply(x, xprev, a);
    }
    memcpy(xprev, x, n * sizeof(aa_float));
    for (aa_int j = 0; j < n; j++) {
      x[j] -= step * Qdiag[j] * xprev[j];
    }
    aa_safeguard(x, xprev, a);
  }
  aa_float err = nrm2_vec(x, n);
  aa_finish(a);
  free(x);
  free(xprev);
  return err;
}

/* aa_init with mem=0 must not crash and both apply/safeguard must be
 * no-ops (this path is what callers use to turn AA off dynamically). */
static const char *test_mem_zero_is_noop(void) {
  AaWork *a = aa_init(10, 0, 1, 1e-8, 1.0, 2.0, 1e10, 0);
  mu_assert("aa_init(mem=0) returned NULL", a != NULL);

  aa_float x[10], xprev[10];
  for (int i = 0; i < 10; i++) {
    x[i] = (aa_float)i * 0.1;
    xprev[i] = x[i];
  }
  aa_float before = nrm2_vec(x, 10);

  aa_float w = aa_apply(x, xprev, a);
  mu_assert("aa_apply(mem=0) should return 0", w == 0);

  aa_int r = aa_safeguard(x, xprev, a);
  mu_assert("aa_safeguard(mem=0) should return 0", r == 0);

  /* No-op means x is untouched. */
  aa_float after = nrm2_vec(x, 10);
  mu_assert("mem=0 apply/safeguard modified inputs", before == after);

  aa_finish(a);
  return 0;
}

static const char *test_dim_zero_rejected(void) {
  AaWork *a = aa_init(0, 1, 1, 1e-8, 1.0, 2.0, 1e10, 0);
  mu_assert("aa_init(dim=0) should return NULL", a == NULL);
  return 0;
}

/* mem=1 is the smallest non-trivial memory — exercises the len=1
 * path of the internal solve. */
static const char *test_mem_one(void) {
  aa_float Qdiag[10];
  for (int i = 0; i < 10; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / 9.0; /* eigs in [0.1, 1] */
  }
  aa_float err = diag_gd(Qdiag, 10, /*step=*/1.0, /*mem=*/1, /*type1=*/1,
                         /*relax=*/1.0, /*iters=*/500, /*seed=*/42);
  mu_assert_less("mem=1 did not converge", err, 1e-4);
  return 0;
}

/* mem > dim is internally capped to dim (for rank stability). Must
 * still produce a working accelerator. */
static const char *test_mem_capped_to_dim(void) {
  aa_float Qdiag[3] = {0.5, 1.0, 0.8};
  aa_float err = diag_gd(Qdiag, 3, /*step=*/1.0, /*mem=*/20, /*type1=*/1,
                         1.0, 400, 0);
  mu_assert_less("mem>dim did not converge", err, 1e-8);
  return 0;
}

/* dim=1 is a degenerate but valid use — make sure nothing assumes n>1. */
static const char *test_dim_one(void) {
  aa_float Qdiag[1] = {0.5};
  aa_float err = diag_gd(Qdiag, 1, /*step=*/2.0, /*mem=*/3, /*type1=*/1,
                         1.0, 50, 0);
  mu_assert_less("dim=1 did not converge", err, 1e-10);
  return 0;
}

/* aa_reset should restore the accelerator to first-iter behavior:
 * running with reset + fresh inputs must produce the same trajectory
 * as a brand-new workspace on the same inputs. */
static const char *test_reset_matches_fresh_init(void) {
  const aa_int n = 5;
  aa_float Qdiag[5] = {0.2, 0.4, 0.6, 0.8, 1.0};
  aa_float step = 1.0;
  aa_float x0[5] = {1, 1, 1, 1, 1};

  AaWork *a = aa_init(n, 3, 1, 1e-10, 1.0, 2.0, 1e10, 0);

  /* First run: 20 iters from x0. */
  aa_float x[5], xprev[5];
  memcpy(x, x0, sizeof(x0));
  for (int i = 0; i < 20; i++) {
    if (i > 0) aa_apply(x, xprev, a);
    memcpy(xprev, x, sizeof(x));
    for (int j = 0; j < n; j++) x[j] -= step * Qdiag[j] * xprev[j];
    aa_safeguard(x, xprev, a);
  }
  aa_float err_first = nrm2_vec(x, n);

  /* Reset and run again from the same x0. */
  aa_reset(a);
  aa_float x2[5], xprev2[5];
  memcpy(x2, x0, sizeof(x0));
  for (int i = 0; i < 20; i++) {
    if (i > 0) aa_apply(x2, xprev2, a);
    memcpy(xprev2, x2, sizeof(x2));
    for (int j = 0; j < n; j++) x2[j] -= step * Qdiag[j] * xprev2[j];
    aa_safeguard(x2, xprev2, a);
  }
  aa_float err_after = nrm2_vec(x2, n);

  aa_finish(a);

  aa_float diff = fabs(err_first - err_after);
  mu_assert_less("reset did not produce identical trajectory", diff, 1e-14);
  return 0;
}

/* reset must clear any "last AA step succeeded" state so a subsequent
 * safeguard call cannot roll inputs back to pre-reset iterates. */
static const char *test_reset_clears_stale_safeguard_state(void) {
  AaWork *a = aa_init(2, 2, 1, 1e-8, 1.0, 1.0, 1e10, 0);
  aa_float x[2] = {1.0, 1.0};
  aa_float f[2] = {0.5, 0.5};

  aa_apply(f, x, a);

  memcpy(x, f, sizeof(x));
  f[0] *= 0.5;
  f[1] *= 0.5;
  aa_apply(f, x, a);

  memcpy(x, f, sizeof(x));
  f[0] *= 0.5;
  f[1] *= 0.5;
  aa_float aa_norm = aa_apply(f, x, a);
  mu_assert("expected a successful AA step before reset", aa_norm > 0);

  aa_reset(a);

  aa_float f_new[2] = {3.0, 4.0};
  aa_float x_new[2] = {1.0, 2.0};
  aa_int safeguard = aa_safeguard(f_new, x_new, a);
  mu_assert("safeguard after reset should be a no-op", safeguard == 0);
  mu_assert("reset should prevent stale rollback of f_new", f_new[0] == 3.0);
  mu_assert("reset should prevent stale rollback of x_new", x_new[0] == 1.0);

  aa_finish(a);
  return 0;
}

/* On a moderately-conditioned problem AA should comfortably beat plain GD
 * at a fixed iteration budget. This is the headline claim of the library,
 * so regress hard. */
static const char *test_aa_accelerates_convergence(void) {
  const aa_int n = 20;
  aa_float Qdiag[20];
  for (int i = 0; i < n; i++) {
    Qdiag[i] = 0.01 + 0.99 * (aa_float)i / 19.0;
  }
  const aa_float step = 1.0; /* = 1 / max(Qdiag) */
  const aa_int iters = 200;
  const unsigned seed = 42;

  aa_float err_no_aa = diag_gd(Qdiag, n, step, /*mem=*/0, 1, 1.0, iters, seed);
  aa_float err_aa = diag_gd(Qdiag, n, step, /*mem=*/5, 1, 1.0, iters, seed);

  printf("  no-AA err = %.3e, AA err = %.3e (ratio %.1fx)\n",
         err_no_aa, err_aa, err_no_aa / err_aa);
  /* Expect AA to beat plain GD by at least 10x on this problem. */
  mu_assert("AA did not beat plain GD by 10x", err_aa * 10 < err_no_aa);
  return 0;
}

/* Exercise the cyclic-buffer path: many iters on small mem means
 * (iter-1) % mem wraps around thousands of times. A stale-index bug
 * would produce wrong results or crash. */
static const char *test_cyclic_buffer_long_run(void) {
  const aa_int n = 5;
  aa_float Qdiag[5] = {0.5, 0.6, 0.7, 0.8, 1.0};
  aa_float err = diag_gd(Qdiag, n, /*step=*/1.0, /*mem=*/3, /*type1=*/1,
                         1.0, /*iters=*/10000, /*seed=*/7);
  mu_assert_less("long run did not converge", err, 1e-12);
  return 0;
}

/* Type-II with mem=1 — symmetric of test_mem_one, verifies both
 * types handle the len=1 solve path. */
static const char *test_mem_one_type2(void) {
  aa_float Qdiag[10];
  for (int i = 0; i < 10; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / 9.0;
  }
  aa_float err = diag_gd(Qdiag, 10, /*step=*/1.0, /*mem=*/1, /*type1=*/0,
                         /*relax=*/1.0, /*iters=*/500, /*seed=*/99);
  mu_assert_less("type-II mem=1 did not converge", err, 1e-4);
  return 0;
}

/* First-iteration behavior: aa_apply on iter 0 must only seed internal
 * state and leave f untouched (return 0). This contract lets callers
 * unconditionally call aa_apply without branching. */
static const char *test_first_iter_is_noop_on_f(void) {
  const aa_int n = 4;
  AaWork *a = aa_init(n, 3, 1, 1e-8, 1.0, 2.0, 1e10, 0);
  aa_float x[4] = {0.1, 0.2, 0.3, 0.4};
  aa_float xprev[4] = {0, 0, 0, 0};
  aa_float snapshot[4];
  memcpy(snapshot, x, sizeof(x));

  aa_float w = aa_apply(x, xprev, a);
  mu_assert("first aa_apply should return weight norm 0", w == 0);
  for (int i = 0; i < n; i++) {
    mu_assert("first aa_apply must not modify f", x[i] == snapshot[i]);
  }

  /* Safeguard before any accepted AA step should also be a no-op. */
  aa_int r = aa_safeguard(x, xprev, a);
  mu_assert("aa_safeguard before any AA step should return 0", r == 0);

  aa_finish(a);
  return 0;
}

/* Long GD past machine-precision convergence. Once iterates saturate, Y's
 * columns are denormal noise and M = Y'Y becomes numerically rank-deficient.
 * gesv can return info=0 with NaN-valued weights; without an isfinite guard
 * those NaNs flow through f -= D*work and poison the iterate. Regression
 * test for that failure mode: after the accelerator has had ample chance
 * to break down, f must still be finite. */
static const char *test_post_convergence_stays_finite(void) {
  const aa_int n = 50;
  const aa_int mem = 5;
  const aa_int iters = 2000;
  aa_float *Qdiag = (aa_float *)malloc(sizeof(aa_float) * n);
  for (aa_int i = 0; i < n; i++) {
    /* eigs in [0.1, 1.0] — well-conditioned so GD drives x to 0 fast. */
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / (aa_float)(n - 1);
  }

  aa_float *x = (aa_float *)malloc(sizeof(aa_float) * n);
  aa_float *xprev = (aa_float *)malloc(sizeof(aa_float) * n);
  for (aa_int i = 0; i < n; i++) {
    x[i] = sin((i + 1) * 0.1); /* deterministic, non-trivial init */
  }

  AaWork *a = aa_init(n, mem, /*type1=*/0, /*reg=*/1e-12, /*relax=*/1.0,
                      /*safeguard=*/2.0, /*max_w=*/1e10, /*verbosity=*/0);
  for (aa_int i = 0; i < iters; i++) {
    if (i > 0) {
      aa_apply(x, xprev, a);
    }
    memcpy(xprev, x, n * sizeof(aa_float));
    for (aa_int j = 0; j < n; j++) {
      x[j] -= 1.0 * Qdiag[j] * xprev[j];
    }
    aa_safeguard(x, xprev, a);
  }

  aa_float err = nrm2_vec(x, n);
  aa_finish(a);
  free(x);
  free(xprev);
  free(Qdiag);

  mu_assert("post-convergence iterate is not finite", isfinite(err));
  return 0;
}

/* =============================================================== */

static const char *all_tests(void) {
  /* Integration tests — GD on random quadratic. */
  printf("type 1, relaxation 1.0\n");
  mu_run_test(gd_type1_relax1);
  printf("type 1, relaxation < 1.0\n");
  mu_run_test(gd_type1_relaxl1);
  printf("type 2, relaxation 1.0\n");
  mu_run_test(gd_type2_relax1);
  printf("type 2, relaxation < 1.0\n");
  mu_run_test(gd_type2_relaxl1);

  /* Unit / edge-case tests. */
  printf("unit: mem=0 is a no-op\n");
  mu_run_test(test_mem_zero_is_noop);
  printf("unit: dim=0 is rejected\n");
  mu_run_test(test_dim_zero_rejected);
  printf("unit: mem=1 works\n");
  mu_run_test(test_mem_one);
  printf("unit: mem=1 works (type-II)\n");
  mu_run_test(test_mem_one_type2);
  printf("unit: mem > dim is capped\n");
  mu_run_test(test_mem_capped_to_dim);
  printf("unit: dim=1 works\n");
  mu_run_test(test_dim_one);
  printf("unit: first aa_apply is a no-op on f\n");
  mu_run_test(test_first_iter_is_noop_on_f);
  printf("unit: aa_reset matches a fresh aa_init\n");
  mu_run_test(test_reset_matches_fresh_init);
  printf("unit: aa_reset clears stale safeguard state\n");
  mu_run_test(test_reset_clears_stale_safeguard_state);
  printf("unit: cyclic buffer survives a long run\n");
  mu_run_test(test_cyclic_buffer_long_run);
  printf("unit: AA accelerates convergence vs plain GD\n");
  mu_run_test(test_aa_accelerates_convergence);
  printf("unit: post-convergence iterates stay finite (NaN guard)\n");
  mu_run_test(test_post_convergence_stays_finite);

  return 0;
}

int main(void) {
  const char *result = all_tests();
  if (result != 0) {
    printf("%s\n", result);
  } else {
    printf("ALL TESTS PASSED\n");
  }
  printf("Tests run: %d\n", tests_run);

  return result != 0;
}
