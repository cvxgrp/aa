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

  aa_float safeguard_factor; /* safeguard tolerance factor */

  aa_float *x;     /* x input to map*/
  aa_float *f;     /* f(x) output of map */
  aa_float *g;     /* x - f(x) */
  aa_float norm_g; /* ||x - f(x)|| */

  /* from previous iteration */
  aa_float *g_prev; /* x_prev - f(x_prev) */

  aa_float *y; /* g - g_prev */
  aa_float *s; /* x - x_prev */
  aa_float *d; /* f - f_prev */

  aa_float *Y; /* matrix of stacked y values */
  aa_float *S; /* matrix of stacked s values */
  aa_float *D; /* matrix of stacked d values = (S-Y) */
  aa_float *M; /* S'Y or Y'Y depending on type of aa */

  /* workspace variables */
  aa_float *work; /* scratch space */

  /* SVD workspace */
  aa_float *Y_work;
  aa_float *sigs;
  aa_float *swork;
  blas_int *iwork;
  blas_int lwork;
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

/* updates the workspace parameters for aa for this iteration */
static void update_accel_params(const aa_float *x, const aa_float *f, AaWork *a,
                                aa_int len) {
  /* at the start a->x = x_prev and a->f = f_prev */
  TIME_TIC
  aa_int idx = (a->iter - 1) % a->mem;
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


static void init_gelsd(AaWork *a) {
  aa_float worksize;
  blas_int bmem = a->mem, bdim = a->dim;
  blas_int neg_one = -1, one = 1, rank, info;
  aa_float neg_onef = -1;
  a->Y_work = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));
  a->sigs = (aa_float *)malloc(a->mem * sizeof(aa_float));
  BLAS(gelsd)(&bdim, &bmem, &one, a->Y, &bdim, a->work, &bdim, a->sigs,
              &neg_onef, &rank, &worksize, &neg_one, &neg_one, &info);
  a->lwork = (blas_int)worksize;
  a->swork = (aa_float *)malloc(a->lwork * sizeof(aa_float));
  a->iwork = (blas_int *)malloc(a->lwork * sizeof(blas_int));
}

/* solves the system of equations to perform the AA update
 * at the end f contains the next iterate to be returned
 */
static aa_float solve_with_gelsd(aa_float *f, AaWork *a, aa_int len) {
  TIME_TIC
  blas_int info = -1, bdim = (blas_int)(a->dim), one = 1, blen = (blas_int)len;
  aa_float onef = 1.0, neg_onef = -1.0, aa_norm;
  blas_int rank;


  memcpy(a->Y_work, a->Y, a->dim * len * sizeof(aa_float));
  memcpy(a->work, a->g, a->dim * sizeof(aa_float));
  BLAS(gelsd)(&bdim, &blen, &one, a->Y_work, &bdim, a->work, &bdim, a->sigs,
              &neg_onef, &rank, a->swork, &a->lwork, a->iwork, &info);

  aa_norm = BLAS(nrm2)(&blen, a->work, &one);
  if (a->verbosity > 1) {
    printf("AA type %i, iter: %i, len %i, info: %i, aa_norm %.2e\n",
           2, (int)a->iter, (int)len, (int)info, aa_norm);
  }

  /* info < 0 input error, input > 0 matrix is singular */
  if (info != 0) {
    if (a->verbosity > 0) {
      printf("Error in AA type %i, iter: %i, len %i, info: %i, aa_norm %.2e\n",
             2, (int)a->iter, (int)len, (int)info, aa_norm);
    }
    a->success = 0;
    /* reset aa for stability */
    aa_reset(a);
    TIME_TOC
    return -aa_norm;
  }

  /* here work = gamma, ie, the correct AA shifted weights */
  /* if solve was successful compute new point */
  /* f -= D * work */
  BLAS(gemv)
  ("NoTrans", &bdim, &blen, &neg_onef, a->D, &bdim, a->work, &one, &onef, f,
   &one);
  a->success = 1; /* this should be the only place we set success = 1 */
  TIME_TOC
  return aa_norm;
}

/*
 * API functions below this line, see aa.h for descriptions.
 */
AaWork *aa_init(aa_int dim, aa_int mem, aa_float safeguard_factor,
                aa_int verbosity) {
  TIME_TIC
  AaWork *a = (AaWork *)calloc(1, sizeof(AaWork));
  if (!a) {
    printf("Failed to allocate memory for AA.\n");
    return (void *)0;
  }
  a->iter = 0;
  a->dim = dim;
  a->mem = mem;
  a->safeguard_factor = safeguard_factor;
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

  a->Y = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));
  a->D = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));

  a->M = (aa_float *)calloc(a->mem * a->mem, sizeof(aa_float));
  a->work = (aa_float *)calloc(MAX(a->mem, a->dim), sizeof(aa_float));
  init_gelsd(a);
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
  /* set various accel quantities */
  update_accel_params(x, f, a, len);

  /* only perform solve steps when the memory is full */
  if (!FILL_MEMORY_BEFORE_SOLVE || a->iter >= a->mem) {
    /* solve linear system, new point overwrites f if successful */
    aa_norm = solve_with_gelsd(f, a, len);
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
    free(a->Y);
    free(a->D);
    free(a->M);
    free(a->Y_work);
    free(a->work);
    free(a->swork);
    free(a->iwork);
    free(a->sigs);
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
