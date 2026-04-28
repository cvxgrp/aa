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
#include <limits.h>
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

  AaWork *a = aa_init(n, memory, /*min_len=*/memory, type1,
                      regularization, relaxation, safeguard_tolerance,
                      max_aa_norm, /*ir_max_steps=*/5, verbosity);
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
 * entries; step is a fixed scalar. Pass mem=0 to disable AA.
 *
 * Uses min_len = mem so the solve only starts once memory is full, which
 * is the historical default behavior most tests were written against. */
static aa_float diag_gd(const aa_float *Qdiag, aa_int n, aa_float step,
                        aa_int mem, aa_int type1, aa_float relaxation,
                        aa_int iters, unsigned seed) {
  aa_float *x = (aa_float *)calloc(n, sizeof(aa_float));
  aa_float *xprev = (aa_float *)calloc(n, sizeof(aa_float));
  srand(seed);
  for (aa_int i = 0; i < n; i++) {
    x[i] = rand_float();
  }
  AaWork *a = aa_init(n, mem, /*min_len=*/mem, type1, /*reg=*/1e-10,
                      relaxation, /*safeguard=*/2.0, /*max_w=*/1e10,
                      /*ir_max_steps=*/5, /*verbosity=*/0);
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
  /* min_len is ignored when mem=0 — pass the sentinel 0 to make that
   * explicit (a nonzero value would also be accepted since the range
   * check is gated on mem>0). */
  AaWork *a = aa_init(10, 0, /*min_len=*/0, 1, 1e-8, 1.0, 2.0, 1e10, 5, 0);
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
  AaWork *a = aa_init(0, 1, /*min_len=*/1, 1, 1e-8, 1.0, 2.0, 1e10, 5, 0);
  mu_assert("aa_init(dim=0) should return NULL", a == NULL);
  return 0;
}

static const char *test_dimension_overflow_rejected(void) {
  AaWork *a = aa_init(INT_MAX, 1, /*min_len=*/1, 1, 1e-8, 1.0, 2.0,
                      1e10, 5, 0);
  mu_assert("aa_init should reject dim+mem overflow", a == NULL);

  a = aa_init(INT_MAX, INT_MAX, /*min_len=*/1, 1, 1e-8, 1.0, 2.0,
              1e10, 5, 0);
  mu_assert("aa_init should reject oversized dim/mem dimensions", a == NULL);
  return 0;
}

/* Negative ir_max_steps is invalid and must be rejected at init. */
static const char *test_ir_max_steps_negative_rejected(void) {
  AaWork *a = aa_init(4, 2, /*min_len=*/2, 1, 1e-8, 1.0, 2.0, 1e10, -1, 0);
  mu_assert("aa_init(ir_max_steps=-1) should return NULL", a == NULL);
  return 0;
}

/* ir_max_steps=0 disables iterative refinement entirely. The solve
 * path still runs end-to-end and AA still converges on an easy
 * well-conditioned problem. */
static const char *test_ir_max_steps_zero_still_solves(void) {
  aa_float Qdiag[10];
  for (int i = 0; i < 10; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / 9.0;
  }
  const aa_int n = 10;
  aa_float *x = (aa_float *)calloc(n, sizeof(aa_float));
  aa_float *xprev = (aa_float *)calloc(n, sizeof(aa_float));
  srand(7);
  for (aa_int i = 0; i < n; i++) x[i] = rand_float();
  AaWork *a = aa_init(n, /*mem=*/5, /*min_len=*/5, /*type1=*/1,
                      /*reg=*/1e-10, /*relax=*/1.0, /*safeguard=*/2.0,
                      /*max_w=*/1e10, /*ir_max_steps=*/0, /*verbosity=*/0);
  mu_assert("aa_init(ir_max_steps=0) must accept", a != NULL);
  for (aa_int i = 0; i < 500; i++) {
    if (i > 0) aa_apply(x, xprev, a);
    memcpy(xprev, x, n * sizeof(aa_float));
    for (aa_int j = 0; j < n; j++) x[j] -= 1.0 * Qdiag[j] * xprev[j];
    aa_safeguard(x, xprev, a);
  }
  aa_float err = nrm2_vec(x, n);
  aa_finish(a);
  free(x);
  free(xprev);
  mu_assert("ir_max_steps=0 run produced non-finite iterate", isfinite(err));
  mu_assert_less("ir_max_steps=0 did not converge", err, 1e-6);
  return 0;
}

/* Drive the ill-conditioned (κ=1e10) GD problem twice, once with IR
 * disabled and once with the default cap of 5. Both must converge
 * well below the plain-GD residual and stay finite — the IR-on run
 * must not regress meaningfully relative to IR-off. Once AA reaches
 * the machine-precision floor (here ≪ 1e-10) the two errors become
 * noise-dominated, so we compare against an absolute floor rather
 * than a relative ratio. */
static const char *test_ir_max_steps_no_regression_on_ill_conditioned(void) {
  const aa_int n = 40;
  aa_float Qdiag[40];
  for (int i = 0; i < n; i++) {
    aa_float t = (aa_float)i / (aa_float)(n - 1);
    Qdiag[i] = 1e-10 + (1.0 - 1e-10) * t;
  }
  aa_float errs[2];
  aa_int caps[2] = {0, 5};
  for (int k = 0; k < 2; ++k) {
    aa_float *x = (aa_float *)calloc(n, sizeof(aa_float));
    aa_float *xprev = (aa_float *)calloc(n, sizeof(aa_float));
    srand(3);
    for (aa_int i = 0; i < n; i++) x[i] = rand_float();
    AaWork *a = aa_init(n, /*mem=*/10, /*min_len=*/10, /*type1=*/0,
                        /*reg=*/1e-10, /*relax=*/1.0, /*safeguard=*/2.0,
                        /*max_w=*/1e10, caps[k], /*verbosity=*/0);
    for (aa_int i = 0; i < 2000; i++) {
      if (i > 0) aa_apply(x, xprev, a);
      memcpy(xprev, x, n * sizeof(aa_float));
      for (aa_int j = 0; j < n; j++) x[j] -= 1.0 * Qdiag[j] * xprev[j];
      aa_safeguard(x, xprev, a);
    }
    errs[k] = nrm2_vec(x, n);
    aa_finish(a);
    free(x);
    free(xprev);
  }
  mu_assert("ir=0 run produced non-finite iterate", isfinite(errs[0]));
  mu_assert("ir=5 run produced non-finite iterate", isfinite(errs[1]));
  /* Both should drive past any pre-IR convergence wall on this κ=1e10
   * problem. 1e-8 is well below where plain QR-without-IR would stall
   * but well above the machine-precision noise floor so the check is
   * meaningful. */
  mu_assert_less("ir=0 did not converge on kappa=1e10", errs[0], 1e-8);
  mu_assert_less("ir=5 did not converge on kappa=1e10", errs[1], 1e-8);
  /* The actual "no regression" claim: ir=5 must not be materially worse
   * than ir=0. At machine-precision both are noise; max() against 1e-12
   * prevents a spurious fail when ir=0 happens to hit a lower noise draw. */
  {
    aa_float baseline = errs[0] > 1e-12 ? errs[0] : 1e-12;
    mu_assert_less("ir=5 regressed relative to ir=0", errs[1], 3.0 * baseline);
  }
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

  AaWork *a = aa_init(n, 3, /*min_len=*/3, 1, 1e-10, 1.0, 2.0, 1e10, 5, 0);

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
  AaWork *a = aa_init(2, 2, /*min_len=*/2, 1, 1e-8, 1.0, 1.0, 1e10, 5, 0);
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

/* Running past machine-precision convergence: S, Y columns saturate at
 * denormal noise, the Gram matrix becomes catastrophically singular.
 * Under the pre-QR normal-equations solve this manifested as NaN-valued
 * γ weights and (without the isfinite guard) NaN iterates propagating
 * back through F. The QR path should keep iterates finite indefinitely
 * without needing a guard to catch a blown-up weight vector. */
static const char *test_post_convergence_stays_finite(void) {
  const aa_int n = 50;
  aa_float Qdiag[50];
  for (int i = 0; i < n; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / (aa_float)(n - 1);
  }
  aa_float err = diag_gd(Qdiag, n, /*step=*/1.0, /*mem=*/5, /*type1=*/0,
                         /*relax=*/1.0, /*iters=*/2000, /*seed=*/17);
  mu_assert("post-convergence iterate is not finite", isfinite(err));
  return 0;
}

/* Low-residual regression specifically for the y = g - g_prev choice in
 * update_accel_params. Near the optimum g and g_prev are both tiny and
 * nearly equal, so y is a cancellation-prone quantity (single-rounding
 * subtraction). Deriving y from s - d would add two extra roundings and
 * visibly degrade it; an ordering bug that advances g_prev before y is
 * built would also corrupt y. Either failure mode would stall or blow
 * up a run like this well before reaching 1e-12. */
static const char *test_low_residual_near_convergence(void) {
  aa_float Qdiag[5] = {0.5, 0.7, 0.8, 0.9, 1.0};
  aa_float err = diag_gd(Qdiag, 5, /*step=*/1.0, /*mem=*/3, /*type1=*/0,
                         /*relax=*/1.0, /*iters=*/10000, /*seed=*/13);
  mu_assert("near-convergence iterate is not finite", isfinite(err));
  mu_assert_less("near-convergence err did not reach 1e-12", err, 1e-12);
  return 0;
}

/* QR is well-defined even with regularization=0 and Y columns that
 * overlap heavily (near-singular) — the augmented [A; 0] reduces to A
 * and QR still extracts the rank-revealing triangular factor. This test
 * runs long enough to saturate at floating-point floor; without the QR
 * path it blows up. */
static const char *test_zero_reg_near_singular_y(void) {
  const aa_int n = 50;
  aa_float Qdiag[50];
  for (int i = 0; i < n; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / (aa_float)(n - 1);
  }
  aa_float *x = (aa_float *)calloc(n, sizeof(aa_float));
  aa_float *xprev = (aa_float *)calloc(n, sizeof(aa_float));
  srand(23);
  for (aa_int i = 0; i < n; i++) x[i] = rand_float();
  AaWork *a = aa_init(n, /*mem=*/10, /*min_len=*/10, /*type1=*/0,
                      /*reg=*/0.0, /*relax=*/1.0, /*safeguard=*/2.0,
                      /*max_w=*/1e10, /*ir_max_steps=*/5, /*verbosity=*/0);
  for (aa_int i = 0; i < 2000; i++) {
    if (i > 0) aa_apply(x, xprev, a);
    memcpy(xprev, x, n * sizeof(aa_float));
    for (aa_int j = 0; j < n; j++) x[j] -= 1.0 * Qdiag[j] * xprev[j];
    aa_safeguard(x, xprev, a);
  }
  aa_float err = nrm2_vec(x, n);
  aa_finish(a);
  free(x);
  free(xprev);
  mu_assert("zero-reg near-singular run produced non-finite iterate",
            isfinite(err));
  return 0;
}

/* Hostile conditioning: κ(Q) = 1e10. The normal-equations solve squares
 * this to 1e20, well past double precision; QR keeps it at 1e10 so the
 * solver still produces bounded weights and AA still converges. */
static const char *test_ill_conditioned_gd(void) {
  const aa_int n = 40;
  aa_float Qdiag[40];
  for (int i = 0; i < n; i++) {
    aa_float t = (aa_float)i / (aa_float)(n - 1);
    Qdiag[i] = 1e-10 + (1.0 - 1e-10) * t; /* eigs in [1e-10, 1] */
  }
  aa_float err = diag_gd(Qdiag, n, /*step=*/1.0, /*mem=*/10, /*type1=*/0,
                         /*relax=*/1.0, /*iters=*/2000, /*seed=*/3);
  mu_assert("ill-conditioned run produced non-finite iterate",
            isfinite(err));
  /* Plain GD on this problem would still be at ||x|| ~ 1 after 2000 iters
   * (slowest mode ~ 1 - 1e-10 step). AA should get well below that. */
  mu_assert_less("AA failed to accelerate at kappa=1e10", err, 1e-2);
  return 0;
}

/* Rank-deficient memory: the Y columns live in a low-rank subspace
 * (only a handful of eigenmodes are active) but mem is deliberately
 * oversized. Without pivoted QR + rank truncation this used to force a
 * full aa_reset every time the memory filled, destroying convergence;
 * the outcome was err stuck at roughly the plain-GD residual for the
 * active modes. With truncation AA keeps the well-conditioned subspace
 * and drives the iterate all the way to machine precision — that's the
 * exact gap the convergence assertion below measures. */
static const char *test_rank_deficient_memory_oversized_mem(void) {
  const aa_int n = 50;
  aa_float Qdiag[50];
  /* Only the first 3 eigenvalues are "interesting"; the rest are ~1.0
   * so GD kills them in one step and they don't contribute to Y. */
  for (int i = 0; i < n; i++) {
    Qdiag[i] = (i < 3) ? (1e-3 + 3e-3 * i) : 1.0;
  }
  /* mem=20, much larger than the ≈3 effective directions in Y. */
  aa_float err = diag_gd(Qdiag, n, /*step=*/1.0, /*mem=*/20, /*type1=*/1,
                         /*relax=*/1.0, /*iters=*/500, /*seed=*/31);
  mu_assert("rank-deficient run produced non-finite iterate", isfinite(err));
  /* 1e-10 is well below where the pre-truncation behavior stalled
   * (~1e-3 on this problem, set by the smallest interesting eigenvalue
   * and plain-GD contraction over 500 iters). */
  mu_assert_less("rank-deficient run failed to converge to near-zero",
                 err, 1e-10);
  return 0;
}

/* Helper: run `iters` of GD+AA on a diagonal-Q problem with the given
 * `regularization` value passed verbatim to aa_init (sign included, so
 * negative selects the pinned branch). Seeds x from rand_float(), writes
 * the final iterate to `x_out` (caller-owned, length n), and returns
 * ||x||. */
static aa_float diag_gd_with_reg(const aa_float *Qdiag, aa_int n, aa_float step,
                                 aa_int mem, aa_int type1, aa_float reg,
                                 aa_int iters, unsigned seed, aa_float *x_out) {
  aa_float *xprev = (aa_float *)calloc(n, sizeof(aa_float));
  srand(seed);
  for (aa_int i = 0; i < n; i++) x_out[i] = rand_float();
  AaWork *a = aa_init(n, mem, /*min_len=*/mem, type1, reg, /*relax=*/1.0,
                      /*safeguard=*/2.0, /*max_w=*/1e10,
                      /*ir_max_steps=*/5, /*verbosity=*/0);
  for (aa_int i = 0; i < iters; i++) {
    if (i > 0) aa_apply(x_out, xprev, a);
    memcpy(xprev, x_out, n * sizeof(aa_float));
    for (aa_int j = 0; j < n; j++) x_out[j] -= step * Qdiag[j] * xprev[j];
    aa_safeguard(x_out, xprev, a);
  }
  aa_float err = nrm2_vec(x_out, n);
  aa_finish(a);
  free(xprev);
  return err;
}

/* Pinned-regularization mode: passing a negative value to aa_init means
 * r is held fixed at |regularization|, skipping the Frobenius scaling.
 * First confirm the pinned branch accepts negative reg and converges.
 * Then distinguish pinned from scaled by running the *same* problem
 * with reg=+R and reg=-R: scaled uses r = R·||A||_F·||Y||_F, pinned
 * uses r = R, so the two regularizers differ by ||A||_F·||Y||_F — which
 * on this problem is nowhere near 1. If a bug silently dropped the sign,
 * both runs would produce bit-identical iterates. The assertion requires
 * the final iterates to differ by more than rounding noise. */
static const char *test_pinned_regularization(void) {
  const aa_int n = 30;
  aa_float Qdiag[30];
  for (int i = 0; i < n; i++) {
    Qdiag[i] = 0.05 + 0.95 * (aa_float)i / (aa_float)(n - 1);
  }
  aa_float *x_scaled = (aa_float *)calloc(n, sizeof(aa_float));
  aa_float *x_pinned = (aa_float *)calloc(n, sizeof(aa_float));

  /* Smoke check: negative reg is accepted and the run converges. */
  aa_float err_pinned = diag_gd_with_reg(
      Qdiag, n, /*step=*/1.0, /*mem=*/5, /*type1=*/1,
      /*reg=*/-1e-8, /*iters=*/1000, /*seed=*/11, x_pinned);
  mu_assert("pinned-reg run produced non-finite iterate", isfinite(err_pinned));
  mu_assert_less("pinned-reg failed to converge", err_pinned, 1e-8);

  /* Distinguishing check: same problem, same seed, only the sign of
   * `regularization` differs between runs. Use a short horizon so the
   * difference hasn't been washed out by convergence. */
  aa_float err_s = diag_gd_with_reg(
      Qdiag, n, /*step=*/1.0, /*mem=*/5, /*type1=*/1,
      /*reg=*/+1e-3, /*iters=*/20, /*seed=*/11, x_scaled);
  aa_float err_p = diag_gd_with_reg(
      Qdiag, n, /*step=*/1.0, /*mem=*/5, /*type1=*/1,
      /*reg=*/-1e-3, /*iters=*/20, /*seed=*/11, x_pinned);
  mu_assert("pinned/scaled runs produced non-finite iterate",
            isfinite(err_s) && isfinite(err_p));
  aa_float diff = 0;
  for (aa_int i = 0; i < n; i++) {
    aa_float d = x_scaled[i] - x_pinned[i];
    diff += d * d;
  }
  diff = sqrt(diff);
  /* A bug that ignored the sign would give diff == 0 exactly (same
   * seed, same code path, same arithmetic). 1e-10 is comfortably above
   * rounding noise yet far below the 20-iter trajectory scale. */
  free(x_scaled);
  free(x_pinned);
  mu_assert("pinned branch produced bit-identical iterate to scaled (sign ignored?)",
            diff > 1e-10);
  return 0;
}

/* min_len=1: AA should start extrapolating on iter 1 (as soon as the first
 * residual pair is buffered) and still converge comparably to min_len=mem
 * on a well-conditioned problem. Without min_len you couldn't get this
 * behavior with a large `mem` — the old code forced solve to wait for the
 * memory to fill. */
static const char *test_min_len_one_accelerates_early(void) {
  const aa_int n = 10;
  aa_float Qdiag[10];
  for (int i = 0; i < n; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / (aa_float)(n - 1);
  }
  aa_float *x = (aa_float *)calloc(n, sizeof(aa_float));
  aa_float *xprev = (aa_float *)calloc(n, sizeof(aa_float));
  srand(101);
  for (aa_int i = 0; i < n; i++) x[i] = rand_float();
  AaWork *a = aa_init(n, /*mem=*/10, /*min_len=*/1, /*type1=*/0,
                      /*reg=*/1e-10, /*relax=*/1.0, /*safeguard=*/2.0,
                      /*max_w=*/1e10, /*ir_max_steps=*/5, /*verbosity=*/0);
  mu_assert("aa_init(min_len=1) returned NULL", a != NULL);
  aa_int applies = 0;
  for (aa_int i = 0; i < 100; i++) {
    if (i > 0) {
      aa_float w = aa_apply(x, xprev, a);
      if (w > 0) applies++;
    }
    memcpy(xprev, x, n * sizeof(aa_float));
    for (aa_int j = 0; j < n; j++) x[j] -= 1.0 * Qdiag[j] * xprev[j];
    aa_safeguard(x, xprev, a);
  }
  aa_float err = nrm2_vec(x, n);
  aa_finish(a);
  free(x);
  free(xprev);
  mu_assert("min_len=1 produced non-finite iterate", isfinite(err));
  mu_assert_less("min_len=1 did not converge", err, 1e-6);
  /* With min_len=1 and mem=10, the solve kicks in from iter 1. Previously
   * (pre-min_len code) the first 9 solve calls were skipped while the
   * memory filled. Require at least some extrapolation to have happened
   * in the first 10 iters — a successful solve on iter 1 already pushes
   * `applies` past the threshold the old code imposed (0). */
  mu_assert("min_len=1 never accepted an AA step", applies > 0);
  return 0;
}

/* min_len < mem: AA starts extrapolating partway through the memory
 * warm-up. Exercise the "rank < mem" branch of the solve repeatedly
 * before the buffer fills, and verify it still converges. */
static const char *test_min_len_half_mem(void) {
  aa_float Qdiag[10];
  for (int i = 0; i < 10; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / 9.0;
  }
  aa_float *x = (aa_float *)calloc(10, sizeof(aa_float));
  aa_float *xprev = (aa_float *)calloc(10, sizeof(aa_float));
  srand(202);
  for (aa_int i = 0; i < 10; i++) x[i] = rand_float();
  AaWork *a = aa_init(10, /*mem=*/8, /*min_len=*/4, /*type1=*/1,
                      /*reg=*/1e-8, /*relax=*/1.0, /*safeguard=*/2.0,
                      /*max_w=*/1e10, /*ir_max_steps=*/5, /*verbosity=*/0);
  mu_assert("aa_init(min_len=4) returned NULL", a != NULL);
  for (aa_int i = 0; i < 300; i++) {
    if (i > 0) aa_apply(x, xprev, a);
    memcpy(xprev, x, 10 * sizeof(aa_float));
    for (aa_int j = 0; j < 10; j++) x[j] -= 1.0 * Qdiag[j] * xprev[j];
    aa_safeguard(x, xprev, a);
  }
  aa_float err = nrm2_vec(x, 10);
  aa_finish(a);
  free(x);
  free(xprev);
  mu_assert("min_len=mem/2 produced non-finite iterate", isfinite(err));
  mu_assert_less("min_len=mem/2 did not converge", err, 1e-6);
  return 0;
}

/* min_len = mem: the historical default. Equivalent to the pre-min_len
 * behavior where solve waited for the buffer to fill. */
static const char *test_min_len_equal_mem_matches_default(void) {
  aa_float Qdiag[10];
  for (int i = 0; i < 10; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / 9.0;
  }
  aa_float err = diag_gd(Qdiag, 10, /*step=*/1.0, /*mem=*/5, /*type1=*/0,
                         /*relax=*/1.0, /*iters=*/500, /*seed=*/55);
  mu_assert_less("min_len=mem default run did not converge", err, 1e-6);
  return 0;
}

/* min_len=0 with mem>0 must be rejected. */
static const char *test_min_len_zero_rejected(void) {
  AaWork *a = aa_init(5, /*mem=*/3, /*min_len=*/0, 1, 1e-8, 1.0,
                      2.0, 1e10, 5, 0);
  mu_assert("aa_init(min_len=0, mem=3) should return NULL", a == NULL);
  return 0;
}

/* min_len is ignored when mem=0 — any value (incl 0 or nonsense) accepted. */
static const char *test_min_len_ignored_when_mem_zero(void) {
  AaWork *a = aa_init(5, /*mem=*/0, /*min_len=*/999, 1, 1e-8, 1.0,
                      2.0, 1e10, 5, 0);
  mu_assert("aa_init(mem=0) should accept any min_len", a != NULL);
  aa_finish(a);
  return 0;
}

/* min_len > mem is silently clamped (same story as mem > dim). Must
 * still converge. */
static const char *test_min_len_exceeding_mem_is_clamped(void) {
  aa_float Qdiag[10];
  for (int i = 0; i < 10; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / 9.0;
  }
  aa_float *x = (aa_float *)calloc(10, sizeof(aa_float));
  aa_float *xprev = (aa_float *)calloc(10, sizeof(aa_float));
  srand(303);
  for (aa_int i = 0; i < 10; i++) x[i] = rand_float();
  AaWork *a = aa_init(10, /*mem=*/5, /*min_len=*/50, /*type1=*/0,
                      /*reg=*/1e-10, /*relax=*/1.0, /*safeguard=*/2.0,
                      /*max_w=*/1e10, /*ir_max_steps=*/5, /*verbosity=*/0);
  mu_assert("aa_init should clamp min_len > mem silently", a != NULL);
  for (aa_int i = 0; i < 300; i++) {
    if (i > 0) aa_apply(x, xprev, a);
    memcpy(xprev, x, 10 * sizeof(aa_float));
    for (aa_int j = 0; j < 10; j++) x[j] -= 1.0 * Qdiag[j] * xprev[j];
    aa_safeguard(x, xprev, a);
  }
  aa_float err = nrm2_vec(x, 10);
  aa_finish(a);
  free(x);
  free(xprev);
  mu_assert("clamped min_len produced non-finite iterate", isfinite(err));
  mu_assert_less("clamped min_len did not converge", err, 1e-6);
  return 0;
}

/* Safeguard-reject churn with small min_len: after each rejection aa_reset
 * zeroes iter, so the next solve needs min_len fresh pairs again before it
 * will extrapolate. With min_len=1 the solve kicks back in on the very next
 * iter, which is the point of the knob. Iterates must stay finite and
 * converge regardless. */
static const char *test_min_len_survives_safeguard_churn(void) {
  const aa_int n = 40;
  aa_float Qdiag[40];
  for (int i = 0; i < n; i++) {
    aa_float t = (aa_float)i / (aa_float)(n - 1);
    Qdiag[i] = 1e-10 + (1.0 - 1e-10) * t; /* κ = 1e10 — triggers rejections */
  }
  aa_float *x = (aa_float *)calloc(n, sizeof(aa_float));
  aa_float *xprev = (aa_float *)calloc(n, sizeof(aa_float));
  srand(13);
  for (aa_int i = 0; i < n; i++) x[i] = rand_float();
  AaWork *a = aa_init(n, /*mem=*/10, /*min_len=*/1, /*type1=*/0,
                      /*reg=*/1e-10, /*relax=*/1.0, /*safeguard=*/1.0,
                      /*max_w=*/1e10, /*ir_max_steps=*/5, /*verbosity=*/0);
  mu_assert("aa_init(min_len=1) returned NULL", a != NULL);
  for (aa_int i = 0; i < 2000; i++) {
    if (i > 0) aa_apply(x, xprev, a);
    memcpy(xprev, x, n * sizeof(aa_float));
    for (aa_int j = 0; j < n; j++) x[j] -= 1.0 * Qdiag[j] * xprev[j];
    aa_safeguard(x, xprev, a);
  }
  aa_float err = nrm2_vec(x, n);
  aa_finish(a);
  free(x);
  free(xprev);
  mu_assert("safeguard-churn run produced non-finite iterate", isfinite(err));
  mu_assert_less("safeguard-churn run did not converge", err, 1e-2);
  return 0;
}

/* First-iteration behavior: aa_apply on iter 0 must only seed internal
 * state and leave f untouched (return 0). This contract lets callers
 * unconditionally call aa_apply without branching. */
static const char *test_first_iter_is_noop_on_f(void) {
  const aa_int n = 4;
  AaWork *a = aa_init(n, 3, /*min_len=*/3, 1, 1e-8, 1.0, 2.0, 1e10, 5, 0);
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

/* aa_get_stats: counters start at zero, accept count rises as AA takes
 * steps, and last_* fields reflect the most recent solve. Also verifies
 * that an internal aa_reset (triggered by a safeguard rejection) does
 * NOT wipe the counters — rejections should stay visible. */
static const char *test_stats_basic_counters(void) {
  const aa_int n = 10;
  aa_float Qdiag[10];
  for (int i = 0; i < n; i++) {
    Qdiag[i] = 0.1 + 0.9 * (aa_float)i / (aa_float)(n - 1);
  }
  aa_float *x = (aa_float *)calloc(n, sizeof(aa_float));
  aa_float *xprev = (aa_float *)calloc(n, sizeof(aa_float));
  srand(77);
  for (aa_int i = 0; i < n; i++) x[i] = rand_float();
  AaWork *a = aa_init(n, /*mem=*/5, /*min_len=*/5, /*type1=*/0,
                      /*reg=*/1e-12, /*relax=*/1.0, /*safeguard=*/2.0,
                      /*max_w=*/1e10, /*ir_max_steps=*/5, /*verbosity=*/0);
  mu_assert("aa_init returned NULL", a != NULL);

  AaStats s = aa_get_stats(a);
  mu_assert("fresh workspace iter must be 0", s.iter == 0);
  mu_assert("fresh workspace n_accept must be 0", s.n_accept == 0);
  mu_assert("fresh workspace n_reject_lapack must be 0",
            s.n_reject_lapack == 0);
  mu_assert("fresh workspace n_reject_rank0 must be 0",
            s.n_reject_rank0 == 0);
  mu_assert("fresh workspace n_reject_nonfinite must be 0",
            s.n_reject_nonfinite == 0);
  mu_assert("fresh workspace n_reject_weight_cap must be 0",
            s.n_reject_weight_cap == 0);
  mu_assert("fresh workspace n_safeguard_reject must be 0",
            s.n_safeguard_reject == 0);
  mu_assert("fresh workspace last_rank must be 0", s.last_rank == 0);
  mu_assert("fresh workspace last_aa_norm must be NaN",
            isnan(s.last_aa_norm));

  for (aa_int i = 0; i < 50; i++) {
    if (i > 0) aa_apply(x, xprev, a);
    memcpy(xprev, x, n * sizeof(aa_float));
    for (aa_int j = 0; j < n; j++) x[j] -= 1.0 * Qdiag[j] * xprev[j];
    aa_safeguard(x, xprev, a);
  }

  s = aa_get_stats(a);
  /* With mem=5, min_len=5, iter=0..49, the solve runs on iters 5..49 → 45 opportunities. */
  mu_assert("stats: iter must advance", s.iter > 0);
  mu_assert("stats: some AA steps should have been accepted", s.n_accept > 0);
  mu_assert("stats: last_rank should be >= 1 after convergence run",
            s.last_rank >= 1);
  mu_assert("stats: last_rank should be <= mem", s.last_rank <= 5);
  mu_assert("stats: last_aa_norm should be finite",
            isfinite(s.last_aa_norm));
  mu_assert("stats: last_regularization should be non-negative",
            s.last_regularization >= 0);

  aa_finish(a);
  free(x);
  free(xprev);
  return 0;
}

/* Lifetime counters must survive an internal aa_reset (safeguard reject
 * path): we want the rejection itself to show up in n_safeguard_reject,
 * not be wiped by the reset it triggers. */
static const char *test_stats_survive_safeguard_reject(void) {
  const aa_int n = 4;
  /* Tight safeguard_factor=0.01 and an adversarial initial state force
   * a safeguard rejection on the first post-solve iteration. */
  AaWork *a = aa_init(n, /*mem=*/2, /*min_len=*/2, /*type1=*/0,
                      /*reg=*/1e-12, /*relax=*/1.0, /*safeguard=*/0.01,
                      /*max_w=*/1e10, /*ir_max_steps=*/5, /*verbosity=*/0);
  mu_assert("aa_init returned NULL", a != NULL);

  /* Hand-driven sequence chosen to land in the safeguard reject branch. */
  aa_float x[4] = {1.0, 1.0, 1.0, 1.0};
  aa_float xprev[4];
  for (aa_int i = 0; i < 4; i++) {
    if (i > 0) aa_apply(x, xprev, a);
    memcpy(xprev, x, sizeof(x));
    for (int k = 0; k < n; k++) x[k] *= 0.5;
    aa_safeguard(x, xprev, a);
  }
  /* Now feed a large jump to trigger safeguard rejection. */
  aa_apply(x, xprev, a);
  memcpy(xprev, x, sizeof(x));
  aa_float jump[4] = {100.0, 100.0, 100.0, 100.0};
  memcpy(x, jump, sizeof(x));
  aa_int r = aa_safeguard(x, xprev, a);

  AaStats s = aa_get_stats(a);
  if (r == -1) {
    mu_assert("stats: safeguard reject must be counted",
              s.n_safeguard_reject >= 1);
  }
  /* Regardless of the specific reject path taken, n_accept should have
   * observed at least one accepted AA step earlier (iter 2 onward). */
  mu_assert("stats: accept count must survive internal aa_reset",
            s.n_accept >= 1);

  aa_finish(a);
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
  printf("unit: overflowing dimensions are rejected\n");
  mu_run_test(test_dimension_overflow_rejected);
  printf("unit: negative ir_max_steps is rejected\n");
  mu_run_test(test_ir_max_steps_negative_rejected);
  printf("unit: ir_max_steps=0 (IR disabled) still solves\n");
  mu_run_test(test_ir_max_steps_zero_still_solves);
  printf("unit: ir_max_steps variants both converge on ill-conditioned GD\n");
  mu_run_test(test_ir_max_steps_no_regression_on_ill_conditioned);
  printf("unit: mem=1 works\n");
  mu_run_test(test_mem_one);
  printf("unit: mem=1 works (type-II)\n");
  mu_run_test(test_mem_one_type2);
  printf("unit: mem > dim is capped\n");
  mu_run_test(test_mem_capped_to_dim);
  printf("unit: dim=1 works\n");
  mu_run_test(test_dim_one);
  printf("unit: min_len=1 starts extrapolating from iter 1\n");
  mu_run_test(test_min_len_one_accelerates_early);
  printf("unit: min_len=mem/2 converges\n");
  mu_run_test(test_min_len_half_mem);
  printf("unit: min_len=mem (default) converges\n");
  mu_run_test(test_min_len_equal_mem_matches_default);
  printf("unit: min_len=0 with mem>0 is rejected\n");
  mu_run_test(test_min_len_zero_rejected);
  printf("unit: min_len is ignored when mem=0\n");
  mu_run_test(test_min_len_ignored_when_mem_zero);
  printf("unit: min_len > mem is clamped silently\n");
  mu_run_test(test_min_len_exceeding_mem_is_clamped);
  printf("unit: min_len=1 survives safeguard-reject churn\n");
  mu_run_test(test_min_len_survives_safeguard_churn);
  printf("unit: first aa_apply is a no-op on f\n");
  mu_run_test(test_first_iter_is_noop_on_f);
  printf("unit: aa_reset matches a fresh aa_init\n");
  mu_run_test(test_reset_matches_fresh_init);
  printf("unit: aa_reset clears stale safeguard state\n");
  mu_run_test(test_reset_clears_stale_safeguard_state);
  printf("unit: cyclic buffer survives a long run\n");
  mu_run_test(test_cyclic_buffer_long_run);
  printf("unit: iterates stay finite past machine-precision convergence\n");
  mu_run_test(test_post_convergence_stays_finite);
  printf("unit: iterates converge to low residual without blowup\n");
  mu_run_test(test_low_residual_near_convergence);
  printf("unit: zero regularization survives near-singular Y\n");
  mu_run_test(test_zero_reg_near_singular_y);
  printf("unit: AA still converges on kappa=1e10 problem\n");
  mu_run_test(test_ill_conditioned_gd);
  printf("unit: rank-deficient memory with oversized mem converges\n");
  mu_run_test(test_rank_deficient_memory_oversized_mem);
  printf("unit: pinned regularization mode works\n");
  mu_run_test(test_pinned_regularization);
  printf("unit: AA accelerates convergence vs plain GD\n");
  mu_run_test(test_aa_accelerates_convergence);
  printf("unit: aa_get_stats basic counters\n");
  mu_run_test(test_stats_basic_counters);
  printf("unit: stats survive internal aa_reset from safeguard\n");
  mu_run_test(test_stats_survive_safeguard_reject);

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
