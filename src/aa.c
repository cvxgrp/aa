/*
 * Anderson acceleration.
 *
 * x: input iterate
 * x_prev: previous input iterate
 * f: f(x) output of map f applied to x
 * g: x - f (error)
 * g_prev: previous error
 * s: x - x_prev
 * y: g - g_prev
 * d: s - y = f - f_prev
 *
 * capital letters are the variables stacked columnwise
 * idx tracks current index where latest quantities written
 * idx cycles from left to right columns in matrix
 *
 * Type-I:
 * return f = f - (S - Y) * ( S'Y + r I)^{-1} ( S'g )
 *
 * Type-II:
 * return f = f - (S - Y) * ( Y'Y + r I)^{-1} ( Y'g )
 *
 */

#include "aa.h"
#include "aa_blas.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define FILL_MEMORY_BEFORE_SOLVE (1)

#if PROFILING > 0

#define TIME_TIC                                                               \
  timer __t;                                                                   \
  tic(&__t);
#define TIME_TOC toc(__func__, &__t);

#include <time.h>
typedef struct timer {
  struct timespec tic;
  struct timespec toc;
} timer;

void tic(timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->tic);
}

aa_float tocq(timer *t) {
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

aa_float toc(const char *str, timer *t) {
  aa_float time = tocq(t);
  printf("%s - time: %8.4f milli-seconds.\n", str, time);
  return time;
}

#else

#define TIME_TIC
#define TIME_TOC

#endif

/* This file uses Anderson acceleration to improve the convergence of
 * a fixed point mapping.
 * At each iteration we need to solve a (small) linear system, we
 * do this using LAPACK ?gesv.
 */

/* contains the necessary parameters to perform aa at each step */
struct ACCEL_WORK {
  aa_int mem;       /* aa memory */
  aa_int dim;       /* variable dimension */
  aa_int iter;      /* current iteration */
  aa_int verbosity; /* verbosity level, 0 is no printing */
  aa_int success;   /* was the last AA step successful or not */

  aa_float safeguard_factor;     /* safeguard tolerance factor */
  aa_int refactorization_period; /* how often to do a full QR re-factor */

  aa_float *x;     /* x input to map*/
  aa_float *f;     /* f(x) output of map */
  aa_float *g;     /* x - f(x) */
  aa_float norm_g; /* ||x - f(x)|| */

  /* from previous iteration */
  aa_float *g_prev; /* x_prev - f(x_prev) */

  aa_float *y; /* g - g_prev */
  aa_float *d; /* f - f_prev */

  aa_float *dy; /* y - y_prev */

  aa_float *Y; /* matrix of stacked y values */
  aa_float *D; /* matrix of stacked d values = (S-Y) */
  aa_float *M; /* S'Y or Y'Y depending on type of aa */

  aa_float *Y_prev;
  aa_float *R;

  /* QR decomp workspace */
  aa_float *tau;
  aa_float *Q;
  blas_int lwork;
  aa_float *qwork;

  /* workspace variables */
  aa_float *work; /* scratch space */
};

/* initialize accel params, in particular x_prev, f_prev, g_prev */
static void init_accel_params(const aa_float *x, const aa_float *f, AaWork *a) {
  TIME_TIC
  blas_int bdim = (blas_int)a->dim;
  aa_float neg_onef = -1.0;
  blas_int one = 1;
  /* x_prev = x */
  memcpy(a->x, x, sizeof(aa_float) * a->dim);
  /* f_prev = f */
  memcpy(a->f, f, sizeof(aa_float) * a->dim);
  /* g_prev = x */
  memcpy(a->g_prev, x, sizeof(aa_float) * a->dim);
  /* g_prev = x_prev - f_prev */
  BLAS(axpy)(&bdim, &neg_onef, f, &one, a->g_prev, &one);
  TIME_TOC
}

static inline aa_int get_idx(AaWork *a) {
  return (a->iter - 1) % a->mem;
}

/* updates the workspace parameters for aa for this iteration */
static void update_accel_params(const aa_float *x, const aa_float *f, AaWork *a,
                                aa_int len) {
  /* at the start a->x = x_prev and a->f = f_prev */
  TIME_TIC
  aa_int idx = get_idx(a);
  blas_int one = 1;
  blas_int bdim = (blas_int)a->dim;
  aa_float neg_onef = -1.0;

  /* g = x */
  memcpy(a->g, x, sizeof(aa_float) * a->dim);
  /* d = f */
  memcpy(a->d, f, sizeof(aa_float) * a->dim);
  /* g =  x - f */
  BLAS(axpy)(&bdim, &neg_onef, f, &one, a->g, &one);
  /* d = f - f_prev */
  BLAS(axpy)(&bdim, &neg_onef, a->f, &one, a->d, &one);

  /* g, d correct here */

  /* y = g */
  memcpy(a->y, a->g, sizeof(aa_float) * a->dim);
  /* y = g - g_prev */
  BLAS(axpy)(&bdim, &neg_onef, a->g_prev, &one, a->y, &one);

  /* y correct here */

  /* Y_prev <- Y */
  memcpy(a->Y_prev, a->Y, a->mem * a->dim * sizeof(aa_float));

  /* dy = y */
  memcpy(a->dy, a->y, sizeof(aa_float) * a->dim);
  /* dy = y - y_prev */
  BLAS(axpy)(&bdim, &neg_onef, &(a->Y[idx * a->dim]), &one, a->dy, &one);

  /* copy y into idx col of Y */
  memcpy(&(a->Y[idx * a->dim]), a->y, sizeof(aa_float) * a->dim);
  /* copy d into idx col of D */
  memcpy(&(a->D[idx * a->dim]), a->d, sizeof(aa_float) * a->dim);

  /* Y, D correct here */

  /* set a->f and a->x for next iter (x_prev and f_prev) */
  memcpy(a->f, f, sizeof(aa_float) * a->dim);
  memcpy(a->x, x, sizeof(aa_float) * a->dim);

  /* x, f correct here */

  memcpy(a->g_prev, a->g, sizeof(aa_float) * a->dim);
  /* g_prev set for next iter here */

  /* compute ||g|| = ||f - x|| */
  a->norm_g = BLAS(nrm2)(&bdim, a->g, &one);

  TIME_TOC
  return;
}

/* y = Q^T x = R^{-T} Y^T x , Where Y = QR */
static void mult_by_q_trans(AaWork *a, aa_float *Y, aa_float *R, aa_float *x,
                            aa_float *y) {
  TIME_TIC
  blas_int one = 1;
  blas_int blen = (blas_int)a->mem;
  blas_int bdim = (blas_int)a->dim;
  aa_float zerof = 0.0;
  aa_float onef = 1.0;
  /* y = Y^T x */
  BLAS(gemv)
  ("Trans", &bdim, &blen, &onef, Y, &bdim, x, &one, &zerof, y, &one);
  /* y = R^{-T} y */
  BLAS(trsv)("Upper", "Trans", "NotUnitDiag", &blen, R, &blen, y, &one);
  TIME_TOC
}

/* solve A w = g  =>  QR w = g  =>  w = R^{-1} Q' g  ( = R^-1 R^{-T} A^T g) */
static aa_float solve(aa_float *f, AaWork *a, aa_int len) {
  TIME_TIC
  blas_int bdim = (blas_int)a->dim;
  blas_int one = 1;
  blas_int blen = (blas_int)len;
  aa_float onef = 1.0;
  aa_float neg_onef = -1.0;
  aa_float *w = a->work;
  aa_float aa_norm;

  /* w = Q' * g */
  mult_by_q_trans(a, a->Y, a->R, a->g, w);
  /* w = R^-1 * work */
  BLAS(trsv)
  ("Upper", "NoTrans", "NotUnitDiag", &blen, a->R, &blen, w, &one);

  aa_norm = BLAS(nrm2)(&blen, a->work, &one);

  /* set f -= D * work */
  BLAS(gemv)
  ("NoTrans", &bdim, &blen, &neg_onef, a->D, &bdim, w, &one, &onef, f, &one);

  a->success = 1; /* this should be the only place we set success = 1 */
  TIME_TOC
  return aa_norm;
}

static void init_qr_workspace(AaWork *a) {
  TIME_TIC
  aa_float worksize;
  blas_int bmem = (blas_int)a->mem;
  blas_int bdim = (blas_int)a->dim;
  blas_int neg_one = -1;
  blas_int info;
  a->tau = (aa_float *)calloc(a->mem, sizeof(aa_float));
  a->Q = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));
  BLAS(geqrf)(&bdim, &bmem, a->Q, &bdim, a->tau, &worksize, &neg_one, &info);
  a->lwork = (blas_int)worksize;
  a->qwork = (aa_float *)calloc(a->lwork, sizeof(aa_float));
  TIME_TOC
}

static void qr_factorize(AaWork *a, aa_int len) {
  TIME_TIC
  aa_int i;
  blas_int blen = (blas_int)len;
  blas_int bdim = (blas_int)a->dim;
  blas_int info;
  memcpy(a->Q, a->Y, sizeof(aa_float) * a->dim * len);
  BLAS(geqrf)(&bdim, &blen, a->Q, &bdim, a->tau, a->qwork, &a->lwork, &info);
  memset(a->R, 0, len * len * sizeof(aa_float));
  for (i = 0; i < len; ++i) {
    memcpy(&(a->R[i * len]), &(a->Q[i * a->dim]), sizeof(aa_float) * (i + 1));
  }
  TIME_TOC
  return;
}

/*
 * Have A = QR (partial, Q'Q = I_n but QQ' != I_m)
 *
 * want \tilde Y = Y + [0 ... dy ... 0] (single column update)
 *
 * First find u, b, r, such that
 *
 * \tilde Y = [Q u] ( [ R ]   + [ 0 ... b ... 0 ] ) = \tilde Q \tilde R
 *                    [ 0 ]     [ 0 ... r ... 0 ]
 *
 *  Under the constraint that ||u|| = 1 and Q'u = 0 (orthogonality) yields
 *
 *    Q b + a u = dy
 *    => Q'(Q b + r u) = Q' dy
 *    => b = Q' dy                    (costs one multiply by Q')
 *
 *    and
 *
 *    r = || dy - Q b|| we can do this without another multiply using:
 *
 *    r^2 = (dy - Qb)' (dy - Qb)
 *        = ||dy||^2 - 2 b'Q'dy + b' Q' Q b
 *        = ||dy||^2 - 2 ||b||^2 + ||b||^2
 *        = ||dy||^2 - ||b||^2
 *
 *  Then we use Givens rotations to make \tilde R upper triangular.
 *
 *  \tilde R starts like:
 *
 *  [ * * * * * * ]
 *  [   * * * * * ]
 *  [     * * * * ]
 *  [     x * * * ]
 *  [     x   * * ]
 *  [     x     * ]
 *  [     r       ]  <- 'fake' bottom row
 *
 *  Using Givens rotations on the rows starting at the bottom row and working up
 *  to get rid of the spike we have upper Hessenberg:
 *
 *  [ * * * * * * ]
 *  [   * * * * * ]
 *  [     * * * * ]
 *  [       * * * ]
 *  [       x * * ]
 *  [         x * ]
 *  [           x ]  <- 'fake' bottom row will end up all zeros
 *
 *  More Givens rotations on the rows starting at the first extra subdiagonal
 *  and working down are used to reduce this to upper triangular.
 *
 *  We don't need or use Q directly, we use A, R^{-1} (fast) instead.
 *
 */
static void update_qr_factorization(AaWork *a) {
  TIME_TIC
  aa_float *R = a->R;
  aa_float *dy = a->dy;
  aa_float *b = a->work;
  blas_int one = 1;
  blas_int bmem = (blas_int)a->mem;
  blas_int bdim = (blas_int)a->dim;
  aa_float onef = 1.0;
  aa_int len = a->mem;
  blas_int blen = (blas_int)a->mem;
  blas_int bn;
  aa_int i;

  /* Givens rotation workspace */
  aa_float nrm_dy, nrm_b;
  aa_float c;
  aa_float s;
  aa_float r1;
  aa_float r2;
  aa_int ridx;
  aa_float r;

  /* get column index into R */
  aa_int idx = get_idx(a);

  /* Y_prev = Q R */

  /* b = Q' * dy - length len */
  /* Use R and Y from previous iteration */
  mult_by_q_trans(a, a->Y_prev, a->R, dy, b);

  /* R col += b */
  BLAS(axpy)(&bmem, &onef, b, &one, &(R[idx * a->mem]), &one);

  nrm_dy = BLAS(nrm2)(&bdim, dy, &one);
  nrm_b = BLAS(nrm2)(&blen, b, &one);

  r = sqrt(MAX(nrm_dy * nrm_dy - nrm_b * nrm_b, 0.));

  /* Now we start the Givens rotations */

  /* Start with fake bottom row of R, extra col of Q */
  ridx = len * (idx + 1) - 1;
  BLAS(rotg)(&(R[ridx]), &r, &c, &s); /* r = garbage after this */
  /* bottom right corner into r */
  r = 0; /* reset r to 0 */
  BLAS(rot)(&one, &(R[len * len - 1]), &one, &r, &one, &c, &s);

  /* r contains bottom right corner here */

  /* Walk up the spike, R finishes upper Hessenberg */

  bn = (blas_int)(len - idx);       /* number of entries in row from spike */
  for (i = len; i > idx + 1; --i) { /* i is row */
    ridx = len * idx + i - 1;
    /* copy values so that the vectors aren't overwritten */
    r1 = R[ridx - 1];
    r2 = R[ridx];
    BLAS(rotg)(&r1, &r2, &c, &s);
    /* note the non-unit stride (inc) here indicates rows */
    BLAS(rot)(&bn, &(R[ridx - 1]), &blen, &(R[ridx]), &blen, &c, &s);
  }

  /* Walk down the sub-diagonal, R finishes upper triangular */
  for (i = idx + 1; i < len - 1; ++i) { /* i is col */
    bn = (blas_int)(len - i); /* number of entries in row from entry */
    ridx = len * i + i;
    /* copy values so that the vectors aren't overwritten */
    r1 = R[ridx];
    r2 = R[ridx + 1];
    BLAS(rotg)(&r1, &r2, &c, &s);
    /* note the non-unit stride (inc) here indicates rows */
    BLAS(rot)(&bn, &(R[ridx]), &blen, &(R[ridx + 1]), &blen, &c, &s);
  }

  /* Finish fake bottom row of R, extra col of Q
   * Note: here we pass the addresses to rotg directly which performs the
   * rotation instead of later calling rot
   */
  BLAS(rotg)(&(R[len * len - 1]), &r, &c, &s); /* r = garbage after this */
  TIME_TOC
  return;
}

/*
 * API functions below this line, see aa.h for descriptions.
 */
AaWork *aa_init(aa_int dim, aa_int mem, aa_float safeguard_factor,
                aa_int refactorization_period, aa_int verbosity) {
  TIME_TIC
  AaWork *a = (AaWork *)calloc(1, sizeof(AaWork));
  if (!a) {
    printf("Failed to allocate memory for AA.\n");
    return (void *)0;
  }
  a->iter = 0;
  a->dim = dim;
  a->mem = MIN(mem, dim); /* otherwise QR low rank and get nans */
  a->safeguard_factor = safeguard_factor;
  a->refactorization_period = refactorization_period;
  a->success = 0;
  a->verbosity = verbosity;
  if (a->mem <= 0) {
    return a;
  }

  a->x = (aa_float *)calloc(a->dim, sizeof(aa_float));
  a->f = (aa_float *)calloc(a->dim, sizeof(aa_float));
  a->g = (aa_float *)calloc(a->dim, sizeof(aa_float));

  a->g_prev = (aa_float *)calloc(a->dim, sizeof(aa_float));

  a->y = (aa_float *)calloc(a->dim, sizeof(aa_float));
  a->d = (aa_float *)calloc(a->dim, sizeof(aa_float));

  a->dy = (aa_float *)calloc(a->dim, sizeof(aa_float));

  a->Y = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));
  a->D = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));

  a->Y_prev = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));

  a->work = (aa_float *)calloc(MAX(a->mem, a->dim), sizeof(aa_float));

  a->R = (aa_float *)calloc(a->mem * a->mem, sizeof(aa_float));

  init_qr_workspace(a);

  TIME_TOC
  return a;
}

aa_float aa_apply(aa_float *f, const aa_float *x, AaWork *a) {
  TIME_TIC
  aa_float aa_norm = 0;
  aa_int len = MIN(a->iter, a->mem);
  a->success = 0; /* if we make an AA step we set this to 1 later */
  if (a->mem <= 0) {
    TIME_TOC
    return aa_norm; /* 0 */
  }
  if (a->iter == 0) {
    /* if first iteration then seed params for next iter */
    init_accel_params(x, f, a);
    a->iter++;
    TIME_TOC
    return aa_norm; /* 0 */
  }
  /* set various acceleration quantities */
  update_accel_params(x, f, a, len);

  if (a->iter >= a->mem) {
    if (a->iter == a->mem) {
      /* initial QR factorization */
      qr_factorize(a, len);
    } else if (a->iter % a->refactorization_period == 0) {
      /* refactorize periodically for stability */
      qr_factorize(a, len);
    } else {
      /* update R factor */
      update_qr_factorization(a);
    }

/* delete this at some point */
#define _TEST (0)
#if _TEST > 0
#define ABS(x) (((x) < 0) ? -(x) : (x))
    aa_float *f_tmp = (aa_float *)malloc(a->dim * sizeof(aa_float));
    memcpy(f_tmp, f, a->dim * sizeof(aa_float));

    aa_float *R_tmp = (aa_float *)malloc(a->mem * a->mem * sizeof(aa_float));
    memcpy(R_tmp, a->R, a->mem * a->mem * sizeof(aa_float));

    aa_norm = solve(f, a, len);

    qr_factorize(a, len);
    aa_float aa_norm_true = solve(f_tmp, a, len);

    memcpy(a->R, R_tmp, a->mem * a->mem * sizeof(aa_float));

    blas_int bdim = (blas_int)a->dim;
    blas_int one = 1;
    aa_float neg_onef = -1.0;

    /* f_tmp -= f */
    BLAS(axpy)(&bdim, &neg_onef, f, &one, f_tmp, &one);
    aa_float nrm_err = BLAS(nrm2)(&bdim, f_tmp, &one);
    if (nrm_err > 1e-6) {
      printf("iter %i\n", a->iter);
      printf("aa_norm %.4e, aa_norm_true %.4e\n", aa_norm, aa_norm_true);
      printf("f error %.4e, f norm %.4e\n", nrm_err,
             BLAS(nrm2)(&bdim, f, &one));
    }
    free(f_tmp);
    free(R_tmp);
#else
    aa_norm = solve(f, a, len);
#endif
  }
  a->iter++;
  TIME_TOC
  return aa_norm;
}

aa_int aa_safeguard(aa_float *f_new, aa_float *x_new, AaWork *a) {
  TIME_TIC
  blas_int bdim = (blas_int)a->dim;
  blas_int one = 1;
  aa_float neg_onef = -1.0;
  aa_float norm_diff;
  if (!a->success) {
    /* last AA update was not successful, no need for safeguarding */
    TIME_TOC
    return 0;
  }
  /* work = x_new */
  memcpy(a->work, x_new, a->dim * sizeof(aa_float));
  /* work = x_new - f_new */
  BLAS(axpy)(&bdim, &neg_onef, f_new, &one, a->work, &one);
  /* norm_diff = || f_new - x_new || */
  norm_diff = BLAS(nrm2)(&bdim, a->work, &one);
  /* g = f - x */
  if (norm_diff > a->safeguard_factor * a->norm_g) {
    /* in this case we reject the AA step and reset */
    memcpy(f_new, a->f, a->dim * sizeof(aa_float));
    memcpy(x_new, a->x, a->dim * sizeof(aa_float));
    if (a->verbosity > 0) {
      printf("AA rejection, iter: %i, norm_diff %.4e, prev_norm_diff %.4e\n",
             (int)a->iter, norm_diff, a->norm_g);
    }
    aa_reset(a);
    TIME_TOC
    return -1;
  }
  TIME_TOC
  return 0;
}

void aa_finish(AaWork *a) {
  if (a) {
    free(a->x);
    free(a->f);
    free(a->g);
    free(a->g_prev);
    free(a->y);
    free(a->d);
    free(a->dy);
    free(a->Y);
    free(a->D);
    free(a->M);
    free(a->work);
    free(a->Y_prev);
    free(a->R);
    free(a->tau);
    free(a->Q);
    free(a->qwork);
    free(a);
  }
  return;
}

void aa_reset(AaWork *a) {
  /* to reset we simply set a->iter = 0 */
  if (a->verbosity > 0) {
    printf("AA reset.\n");
  }
  a->iter = 0;
  return;
}
