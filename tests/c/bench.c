/* Latency and accuracy benchmarks for libaa.
 *
 * Prints a table of per-configuration numbers every time `make test` runs,
 * so CI logs carry a baseline we can eyeball for regressions. Wall times
 * on CI runners jitter from run to run — the counters (applies, aa_rej,
 * sg_rej) and final_err are the stable regression signals.
 *
 * Problem: gradient descent on a diagonal convex quadratic
 *   (1/2) x^T Q x,  Q = diag(eigs),  eigs uniform in [1/cond, 1],
 *   F(x) = x - step * (Qdiag .* x),  step = 1/max(eigs) = 1,
 *   optimum at x = 0,  initial x_i = sin((i+1) * 0.1).
 *
 * The map is intentionally cheap (O(dim)) so AA overhead shows up as a
 * meaningful fraction of wall time.
 *
 * Sweeps:
 *   scan:dim        fixed mem, varies dim        (shows AA scaling in dim)
 *   scan:mem        fixed dim, varies mem        (shows QR scaling in mem)
 *   scan:type       type-I vs type-II, same cfg
 *   scan:cond       varies conditioning          (convergence pressure)
 *   relaxation      relaxation != 1.0 path
 *   near-optimum    long run on an easy problem — after ~20 iters the
 *                   iterates are at machine precision, so S,Y columns
 *                   are denormal noise and A_aug is severely
 *                   rank-deficient. Exercises the pivoted-QR rank
 *                   truncation path (most iters) and the aa_reset
 *                   fallback (when truncation yields rank 0). Regressions
 *                   in the numerics show up here as blown-up final_err,
 *                   NaNs, or a change in the aa_rej/sg_rej counts.
 *   noisy-floor     deterministic but iteration-dependent perturbation
 *                   injected into F(x). Iterates never fully converge —
 *                   they bounce within a ball of radius ~ noise_scale.
 *                   This is the *realistic* near-optimum regime (the
 *                   ~1e-6 iterate-difference world that real solvers
 *                   live in: stochastic gradients, approximate prox
 *                   operators, fixed tolerance on inner solves). A_aug
 *                   is ill-conditioned but not rank-collapsed; it's
 *                   the hardest case for AA numerics because the signal
 *                   is O(noise_scale) and the noise is O(eps·|x|).
 *
 * Columns:
 *   final_err  final ||x|| (optimum is 0); accuracy signal
 *   applies    # successful aa_apply calls (returned positive weight norm)
 *   aa_rej     # aa_apply calls that were rejected (returned <= 0)
 *   sg_rej     # aa_safeguard calls that rolled the step back
 *   total_ms   total wall-clock for the whole run (map + AA)
 *   us/apply   average microseconds per aa_apply + aa_safeguard call
 *   aa%        fraction of wall time spent inside AA
 */
#include "aa.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
typedef struct _timer {
  LARGE_INTEGER tic;
  LARGE_INTEGER toc;
} _timer;

static void _tic(_timer *t) {
  QueryPerformanceCounter(&t->tic);
}

static aa_float _tocq(_timer *t) {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&t->toc);
  return (aa_float)(t->toc.QuadPart - t->tic.QuadPart) /
         (aa_float)freq.QuadPart * 1e3;
}
#else
typedef struct _timer {
  struct timespec tic;
  struct timespec toc;
} _timer;

static void _tic(_timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->tic);
}

static aa_float _tocq(_timer *t) {
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

typedef struct {
  const char *label;
  aa_int dim;
  aa_int mem;
  aa_int type1;
  aa_float relaxation;
  aa_float cond;         /* condition number; min eig = 1/cond, max = 1.0 */
  aa_int iters;
  aa_float regularization;
  aa_float noise_scale;  /* 0 = clean; else iter-dependent perturbation
                          * added to F(x), simulating a noisy fixed-point
                          * map. Iterates settle in a ball of ~noise_scale. */
} bench_cfg;

static aa_float nrm2(const aa_float *x, aa_int n) {
  aa_float s = 0;
  for (aa_int i = 0; i < n; i++) s += x[i] * x[i];
  return sqrt(s);
}

static void run_one(bench_cfg cfg) {
  aa_int n = cfg.dim;
  aa_float *x = (aa_float *)malloc(sizeof(aa_float) * n);
  aa_float *xprev = (aa_float *)malloc(sizeof(aa_float) * n);
  aa_float *Qdiag = (aa_float *)malloc(sizeof(aa_float) * n);

  /* Reproducible problem setup — no RNG state, identical every run. */
  aa_float eig_lo = 1.0 / cfg.cond;
  for (aa_int i = 0; i < n; i++) {
    aa_float t = (n == 1) ? 0 : (aa_float)i / (aa_float)(n - 1);
    Qdiag[i] = eig_lo + (1.0 - eig_lo) * t;
    x[i] = sin((aa_float)(i + 1) * 0.1); /* reproducible, nonzero */
  }
  aa_float step = 1.0; /* = 1 / max(Qdiag) */

  AaWork *a = aa_init(n, cfg.mem, /*min_len=*/cfg.mem, cfg.type1,
                      cfg.regularization, cfg.relaxation,
                      /*safeguard_factor=*/2.0,
                      /*max_weight_norm=*/1e10, /*ir_max_steps=*/5,
                      /*verbosity=*/0);
  if (!a) {
    printf("%-20s | aa_init failed\n", cfg.label);
    free(x); free(xprev); free(Qdiag);
    return;
  }

  _timer total_timer, aa_timer;
  aa_float aa_ms = 0.0;
  aa_int applies = 0, aa_rej = 0, sg_rej = 0;

  _tic(&total_timer);
  for (aa_int i = 0; i < cfg.iters; i++) {
    if (i > 0) {
      _tic(&aa_timer);
      aa_float w = aa_apply(x, xprev, a);
      aa_ms += _tocq(&aa_timer);
      if (w > 0) applies++;
      else       aa_rej++;
    }
    memcpy(xprev, x, sizeof(aa_float) * n);
    /* x = F(xprev): diagonal GD, optionally with iter-dependent noise
     * scaled to ~noise_scale so iterates keep moving by O(noise_scale). */
    if (cfg.noise_scale > 0) {
      for (aa_int j = 0; j < n; j++) {
        aa_float w = sin((aa_float)(i * 131 + j + 1) * 0.07);
        x[j] -= step * Qdiag[j] * xprev[j];
        x[j] += cfg.noise_scale * w;
      }
    } else {
      for (aa_int j = 0; j < n; j++) {
        x[j] -= step * Qdiag[j] * xprev[j];
      }
    }
    _tic(&aa_timer);
    aa_int sg = aa_safeguard(x, xprev, a);
    aa_ms += _tocq(&aa_timer);
    if (sg != 0) sg_rej++;
  }
  aa_float total_ms = _tocq(&total_timer);
  aa_float err = nrm2(x, n);

  aa_int aa_calls = applies + aa_rej;
  aa_float us_per_apply = (aa_calls > 0) ? (aa_ms * 1e3 / aa_calls) : 0.0;
  aa_float aa_frac = (total_ms > 0) ? (aa_ms / total_ms) : 0.0;

  printf("%-20s | %6d %4d   %-2s  %4.2f | %6d | %10.3e | %6d %5d %5d | %9.2f | %9.2f | %5.1f%%\n",
         cfg.label, (int)cfg.dim, (int)cfg.mem,
         cfg.type1 ? "I" : "II", (double)cfg.relaxation, (int)cfg.iters,
         err, (int)applies, (int)aa_rej, (int)sg_rej,
         total_ms, us_per_apply, aa_frac * 100.0);

  aa_finish(a);
  free(x);
  free(xprev);
  free(Qdiag);
}

static void print_header(const char *section) {
  printf("\n-- %s --\n", section);
  printf("%-20s | %6s %4s %4s %4s | %6s | %10s | %6s %5s %5s | %9s | %9s | %6s\n",
         "label", "dim", "mem", "type", "rlx", "iters",
         "final_err", "applied", "a_rej", "s_rej",
         "total_ms", "us/call", "aa%");
  printf("------------------------------------------------------------------"
         "-------------------------------------------------------------------"
         "---\n");
}

int main(void) {
  /* Warmup: first BLAS call pays dynamic-linker / kernel fault costs that
   * would otherwise get charged to whichever config runs first. Throw away
   * a small run to amortize that. */
  {
    const aa_int wd = 50, wm = 5, wi = 50;
    aa_float *x = (aa_float *)malloc(sizeof(aa_float) * wd);
    aa_float *xprev = (aa_float *)malloc(sizeof(aa_float) * wd);
    for (aa_int i = 0; i < wd; i++) { x[i] = 1.0; xprev[i] = 0.0; }
    AaWork *a = aa_init(wd, wm, /*min_len=*/wm, 0, 1e-12, 1.0, 2.0, 1e10, 5, 0);
    for (aa_int i = 0; i < wi; i++) {
      if (i > 0) aa_apply(x, xprev, a);
      memcpy(xprev, x, sizeof(aa_float) * wd);
      for (aa_int j = 0; j < wd; j++) x[j] *= 0.9;
      aa_safeguard(x, xprev, a);
    }
    aa_finish(a);
    free(x); free(xprev);
  }

  printf("\n=== AA C benchmarks (GD on diagonal quadratic) ===\n");
  printf("optimum is x=0; final_err is ||x|| after `iters` iterations.\n");

  /* -------- Scan: dim --------
   * cond=1e6 + iters=300 keeps every size in the "still making progress"
   * regime so the timing comparison reflects productive AA work, not
   * degenerate post-convergence spinning. Both types are swept because
   * type-I is the dominant deployment for this library and carries
   * extra per-iter cost in the QR path (second ormqr + gesv). */
  print_header("scan:dim type-II (fixed mem=10, cond=1e6)");
  bench_cfg dim_sweep_ii[] = {
      {"dim=10",      10,    10, 0, 1.0, 1e6, 300, 1e-12, 0.0},
      {"dim=100",     100,   10, 0, 1.0, 1e6, 300, 1e-12, 0.0},
      {"dim=1000",    1000,  10, 0, 1.0, 1e6, 300, 1e-12, 0.0},
      {"dim=5000",    5000,  10, 0, 1.0, 1e6, 300, 1e-12, 0.0},
      {"dim=20000",   20000, 10, 0, 1.0, 1e6, 300, 1e-12, 0.0},
  };
  for (size_t i = 0; i < sizeof(dim_sweep_ii)/sizeof(*dim_sweep_ii); i++) run_one(dim_sweep_ii[i]);

  print_header("scan:dim type-I  (fixed mem=10, cond=1e6)");
  bench_cfg dim_sweep_i[] = {
      {"dim=10",      10,    10, 1, 1.0, 1e6, 300, 1e-8, 0.0},
      {"dim=100",     100,   10, 1, 1.0, 1e6, 300, 1e-8, 0.0},
      {"dim=1000",    1000,  10, 1, 1.0, 1e6, 300, 1e-8, 0.0},
      {"dim=5000",    5000,  10, 1, 1.0, 1e6, 300, 1e-8, 0.0},
      {"dim=20000",   20000, 10, 1, 1.0, 1e6, 300, 1e-8, 0.0},
  };
  for (size_t i = 0; i < sizeof(dim_sweep_i)/sizeof(*dim_sweep_i); i++) run_one(dim_sweep_i[i]);

  /* -------- Scan: mem (QR solve is O(dim * mem^2) so mem growth shows
   * up clearly) -------- */
  print_header("scan:mem type-II (fixed dim=500, cond=1e4)");
  bench_cfg mem_sweep_ii[] = {
      {"mem=1",       500, 1,   0, 1.0, 1e4, 500, 1e-12, 0.0},
      {"mem=5",       500, 5,   0, 1.0, 1e4, 500, 1e-12, 0.0},
      {"mem=10",      500, 10,  0, 1.0, 1e4, 500, 1e-12, 0.0},
      {"mem=20",      500, 20,  0, 1.0, 1e4, 500, 1e-12, 0.0},
      {"mem=50",      500, 50,  0, 1.0, 1e4, 500, 1e-12, 0.0},
      {"mem=100",     500, 100, 0, 1.0, 1e4, 500, 1e-12, 0.0},
  };
  for (size_t i = 0; i < sizeof(mem_sweep_ii)/sizeof(*mem_sweep_ii); i++) run_one(mem_sweep_ii[i]);

  print_header("scan:mem type-I  (fixed dim=500, cond=1e4)");
  bench_cfg mem_sweep_i[] = {
      {"mem=1",       500, 1,   1, 1.0, 1e4, 500, 1e-8, 0.0},
      {"mem=5",       500, 5,   1, 1.0, 1e4, 500, 1e-8, 0.0},
      {"mem=10",      500, 10,  1, 1.0, 1e4, 500, 1e-8, 0.0},
      {"mem=20",      500, 20,  1, 1.0, 1e4, 500, 1e-8, 0.0},
      {"mem=50",      500, 50,  1, 1.0, 1e4, 500, 1e-8, 0.0},
      {"mem=100",     500, 100, 1, 1.0, 1e4, 500, 1e-8, 0.0},
  };
  for (size_t i = 0; i < sizeof(mem_sweep_i)/sizeof(*mem_sweep_i); i++) run_one(mem_sweep_i[i]);

  /* -------- Scan: type -------- */
  print_header("scan:type (fixed dim=500, mem=10, cond=1e4)");
  bench_cfg type_sweep[] = {
      {"type-I  reg=1e-8",  500, 10, 1, 1.0, 1e4, 1000, 1e-8,  0.0},
      {"type-I  reg=1e-12", 500, 10, 1, 1.0, 1e4, 1000, 1e-12, 0.0},
      {"type-II reg=1e-12", 500, 10, 0, 1.0, 1e4, 1000, 1e-12, 0.0},
      {"type-II reg=0",     500, 10, 0, 1.0, 1e4, 1000, 0.0,   0.0},
  };
  for (size_t i = 0; i < sizeof(type_sweep)/sizeof(*type_sweep); i++) run_one(type_sweep[i]);

  /* -------- Scan: cond -------- */
  print_header("scan:cond type-II (fixed dim=500, mem=10)");
  bench_cfg cond_sweep_ii[] = {
      {"cond=10",    500, 10, 0, 1.0, 10,   500,  1e-12, 0.0},
      {"cond=100",   500, 10, 0, 1.0, 100,  500,  1e-12, 0.0},
      {"cond=1e4",   500, 10, 0, 1.0, 1e4,  500,  1e-12, 0.0},
      {"cond=1e6",   500, 10, 0, 1.0, 1e6,  2000, 1e-12, 0.0},
      {"cond=1e8",   500, 10, 0, 1.0, 1e8,  2000, 1e-12, 0.0},
  };
  for (size_t i = 0; i < sizeof(cond_sweep_ii)/sizeof(*cond_sweep_ii); i++) run_one(cond_sweep_ii[i]);

  print_header("scan:cond type-I  (fixed dim=500, mem=10)");
  bench_cfg cond_sweep_i[] = {
      {"cond=10",    500, 10, 1, 1.0, 10,   500,  1e-8, 0.0},
      {"cond=100",   500, 10, 1, 1.0, 100,  500,  1e-8, 0.0},
      {"cond=1e4",   500, 10, 1, 1.0, 1e4,  500,  1e-8, 0.0},
      {"cond=1e6",   500, 10, 1, 1.0, 1e6,  2000, 1e-8, 0.0},
      {"cond=1e8",   500, 10, 1, 1.0, 1e8,  2000, 1e-8, 0.0},
  };
  for (size_t i = 0; i < sizeof(cond_sweep_i)/sizeof(*cond_sweep_i); i++) run_one(cond_sweep_i[i]);

  /* -------- Relaxation (exercises x_work path) -------- */
  print_header("relaxation type-II (fixed dim=500, mem=10, cond=1e4)");
  bench_cfg relax_sweep_ii[] = {
      {"relax=1.0 (none)",  500, 10, 0, 1.0,  1e4, 1000, 1e-12, 0.0},
      {"relax=0.95",        500, 10, 0, 0.95, 1e4, 1000, 1e-12, 0.0},
      {"relax=1.2 (over)",  500, 10, 0, 1.2,  1e4, 1000, 1e-12, 0.0},
  };
  for (size_t i = 0; i < sizeof(relax_sweep_ii)/sizeof(*relax_sweep_ii); i++) run_one(relax_sweep_ii[i]);

  print_header("relaxation type-I  (fixed dim=500, mem=10, cond=1e4)");
  bench_cfg relax_sweep_i[] = {
      {"relax=1.0 (none)",  500, 10, 1, 1.0,  1e4, 1000, 1e-8, 0.0},
      {"relax=0.95",        500, 10, 1, 0.95, 1e4, 1000, 1e-8, 0.0},
      {"relax=1.2 (over)",  500, 10, 1, 1.2,  1e4, 1000, 1e-8, 0.0},
  };
  for (size_t i = 0; i < sizeof(relax_sweep_i)/sizeof(*relax_sweep_i); i++) run_one(relax_sweep_i[i]);

  /* -------- Near-optimum: machine-precision saturation ------------
   * Easy well-conditioned problems where AA converges to machine
   * precision within ~20 iters; the remaining iterations run on
   * iterates of magnitude O(eps). S, Y columns are denormal noise
   * and A_aug is severely rank-deficient. This is the degenerate
   * extreme — most iters land in the rank-truncation path and the
   * occasional rank-0 case triggers aa_reset.
   */
  print_header("near-optimum: long runs past machine-precision convergence");
  bench_cfg near_opt[] = {
      {"saturate dim=50",   50,  5,  0, 1.0, 10, 2000, 1e-12, 0.0},
      {"saturate dim=200",  200, 10, 0, 1.0, 10, 2000, 1e-12, 0.0},
      {"mem=20 tail",       200, 20, 0, 1.0, 10, 2000, 1e-12, 0.0},
      {"mem=50 tail",       200, 50, 0, 1.0, 10, 2000, 1e-12, 0.0},
      {"type-I saturate",   200, 10, 1, 1.0, 10, 2000, 1e-8,  0.0},
      /* Zero regularization amplifies ill-conditioning — a canary for
       * the rank-truncation + aa_reset fallback. */
      {"no-reg saturate",   200, 10, 0, 1.0, 10, 2000, 0.0,   0.0},
  };
  for (size_t i = 0; i < sizeof(near_opt)/sizeof(*near_opt); i++) run_one(near_opt[i]);

  /* -------- Near-optimum: the realistic noisy regime ------------
   * Iterates never reach machine precision — they bounce inside a
   * ball of radius ~noise_scale. Consecutive iterate differences
   * are O(noise_scale), not O(eps). This mimics what real solvers
   * see: stochastic gradients, approximate prox maps, inner solves
   * with fixed tolerance. A_aug is ill-conditioned (signal is
   * O(noise_scale), noise is O(eps·|x|)) but not rank-collapsed —
   * which is actually harder for AA than the degenerate extreme
   * above, because the regularization has to be tuned for it but
   * the rank-truncation safety net rarely fires.
   */
  print_header("noisy-floor: realistic near-optimum regime, iter diffs ~ noise_scale");
  bench_cfg noisy[] = {
      /* noise_scale = 1e-4: moderate — easy for AA to track. */
      {"noise=1e-4 type-II", 200, 10, 0, 1.0, 10, 2000, 1e-12, 1e-4},
      {"noise=1e-4 type-I",  200, 10, 1, 1.0, 10, 2000, 1e-8,  1e-4},
      /* noise_scale = 1e-6: the regime the user pointed out — iterate
       * differences of O(1e-6), cancellation in y = g-g_prev is severe
       * (signal 1e-6, input magnitudes 1e-6, roundoff 1e-22). */
      {"noise=1e-6 type-II", 200, 10, 0, 1.0, 10, 2000, 1e-12, 1e-6},
      {"noise=1e-6 type-I",  200, 10, 1, 1.0, 10, 2000, 1e-8,  1e-6},
      {"noise=1e-6 mem=20",  200, 20, 0, 1.0, 10, 2000, 1e-12, 1e-6},
      /* noise_scale = 1e-8: tight — iterate differences of O(1e-8)
       * with double precision gives ~8 digits of signal into the QR.
       * Stress test. */
      {"noise=1e-8 type-II", 200, 10, 0, 1.0, 10, 2000, 1e-12, 1e-8},
      {"noise=1e-8 type-I",  200, 10, 1, 1.0, 10, 2000, 1e-8,  1e-8},
      /* Wider problems in the noisy regime: gemv dimension matters. */
      {"noisy wide dim=2000",2000,10, 0, 1.0, 10, 1000, 1e-12, 1e-6},
      /* no regularization in the noisy regime — this should be the
       * most numerically hostile config in the whole bench. Expect
       * many aa_rej / sg_rej events. */
      {"noisy no-reg",       200, 10, 0, 1.0, 10, 2000, 0.0,   1e-6},
  };
  for (size_t i = 0; i < sizeof(noisy)/sizeof(*noisy); i++) run_one(noisy[i]);

  printf("\n");
  return 0;
}
