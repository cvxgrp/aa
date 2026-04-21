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
 * Both types reduce to the same regularized least-squares augmentation
 *     (A'B + r I) γ = A' g
 *     ⇔   [A; √r I]' [B; √r I] γ = [A; √r I]' [g; 0],
 * where A = S (type-I) or A = Y (type-II), and B = Y. We solve via a thin
 * QR factorization of the augmented A, which keeps the conditioning at
 * κ(A_aug) rather than the κ(A_aug)² that a normal-equations solve would
 * incur — critical near the optimum where Y rows are tiny and the Gram
 * matrix becomes numerically singular.
 */

#include <math.h>

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

#ifdef _WIN32
#include <windows.h>
typedef struct timer {
  LARGE_INTEGER tic;
  LARGE_INTEGER toc;
} timer;

void tic(timer *t) {
  QueryPerformanceCounter(&t->tic);
}

aa_float tocq(timer *t) {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&t->toc);
  return (aa_float)(t->toc.QuadPart - t->tic.QuadPart) / (aa_float)freq.QuadPart * 1e3;
}
#else
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
#endif

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
  aa_int type1;     /* bool, if true type 1 aa otherwise type 2 */
  aa_int mem;       /* aa memory */
  aa_int dim;       /* variable dimension */
  aa_int iter;      /* current iteration */
  aa_int verbosity; /* verbosity level, 0 is no printing */
  aa_int success;   /* was the last AA step successful or not */

  aa_float relaxation;       /* relaxation x and f, beta in some papers */
  aa_float regularization;   /* regularization */
  aa_float safeguard_factor; /* safeguard tolerance factor */
  aa_float max_weight_norm;  /* maximum norm of AA weights */

  aa_float *x;     /* x input to map*/
  aa_float *f;     /* f(x) output of map */
  aa_float *g;     /* x - f(x) */
  aa_float norm_g; /* ||x - f(x)|| */

  /* from previous iteration */
  aa_float *g_prev; /* x_prev - f(x_prev) */

  aa_float *Y; /* matrix of stacked y values */
  aa_float *S; /* matrix of stacked s values */
  aa_float *D; /* matrix of stacked d values = (S-Y) */

  /* QR workspaces, sized for the augmented problem. */
  aa_float *A_aug;   /* (dim + mem) x mem  -- [A; √r I]; factored in place */
  aa_float *B_aug;   /* (dim + mem) x mem  -- [Y; √r I] (type-I only) */
  aa_float *c_aug;   /* (dim + mem)        -- [g; 0], overwritten by Q' c */
  aa_float *tau;     /* mem                -- Householder scalars */
  aa_float *qr_work; /* lwork              -- LAPACK scratch for geqrf/ormqr */
  blas_int qr_lwork; /* size of qr_work, chosen via workspace query at init */

  aa_float *W;    /* mem x mem scratch: Q' B_aug top block (type-I gesv) */
  blas_int *ipiv; /* gesv permutation (type-I) */

  /* dim-sized scratch used by aa_safeguard for the x_new - f_new diff. */
  aa_float *work;

  aa_float *x_work; /* workspace (= x) for when relaxation != 1.0 */
};

/* Tikhonov regularization scaled with the problem. Matches the prior
 * behavior's intent (r grows with the magnitude of A'B so `regularization`
 * stays unitless), but uses the cheap Frobenius-norm upper bound
 *     ||A'B||_F ≤ ||A||_F · ||B||_F
 * instead of maintaining a Gram matrix. For type-II A == B so this is
 * ||Y||_F², the same scale as the previous ||Y'Y||_F up to a factor
 * ≤ √mem. */
static aa_float compute_regularization(AaWork *a, aa_int len) {
  TIME_TIC
  blas_int bmat = (blas_int)(a->dim * len), one = 1;
  aa_float nrm_a = BLAS(nrm2)(&bmat, a->type1 ? a->S : a->Y, &one);
  aa_float nrm_y = a->type1 ? BLAS(nrm2)(&bmat, a->Y, &one) : nrm_a;
  aa_float r = a->regularization * nrm_a * nrm_y;
  if (a->verbosity > 2) {
    printf("iter: %i, ||A||_F %.2e, ||Y||_F %.2e, r: %.2e\n",
           (int)a->iter, nrm_a, nrm_y, r);
  }
  TIME_TOC
  return r;
}

/* Build [M; √r I_len] column-major into `dst` with fixed leading dim
 * (dim + mem). When len < mem we zero-pad the unused trailing rows so
 * the QR factorization still operates on a well-defined (dim+mem) x len
 * block; the zero rows don't change the solve. */
static void build_augmented(aa_float *dst, const aa_float *src, aa_int dim,
                            aa_int mem, aa_int len, aa_float sqrt_r) {
  aa_int i;
  aa_int aug_rows = dim + mem;
  for (i = 0; i < len; ++i) {
    aa_float *col = &dst[i * aug_rows];
    memcpy(col, &src[i * dim], dim * sizeof(aa_float));
    memset(&col[dim], 0, mem * sizeof(aa_float));
    col[dim + i] = sqrt_r;
  }
}

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

/* updates the workspace parameters for aa for this iteration
 *
 * Writes this iteration's s, d, y columns directly into S, D, Y at slot
 * `idx` — no intermediate scratch. Numerically sensitive because:
 *
 *  - y is computed as g - g_prev (ONE rounding into a cancellation-prone
 *    quantity). Deriving y from s - d would add two extra roundings and
 *    make y noticeably worse near convergence where g and g_prev are
 *    tiny and nearly equal.
 *
 *  - The reads of a->x, a->f, a->g_prev all require the PREVIOUS
 *    iteration's values, so state advance (x_prev <- x, f_prev <- f,
 *    g_prev <- g) must happen AFTER everything that reads them. s uses
 *    old a->x; d uses old a->f; y uses old a->g_prev. */
static void update_accel_params(const aa_float *x, const aa_float *f, AaWork *a,
                                aa_int len) {
  /* Entry invariant:  a->x == x_prev, a->f == f_prev, a->g_prev == g_prev. */
  TIME_TIC
  aa_int idx = (a->iter - 1) % a->mem;
  blas_int one = 1;
  blas_int bdim = (blas_int)a->dim;
  aa_float neg_onef = -1.0;
  aa_float *s_col = &(a->S[idx * a->dim]);
  aa_float *d_col = &(a->D[idx * a->dim]);
  aa_float *y_col = &(a->Y[idx * a->dim]);

  /* S[:, idx] = x - x_prev  (reads old a->x). */
  memcpy(s_col, x, sizeof(aa_float) * a->dim);
  BLAS(axpy)(&bdim, &neg_onef, a->x, &one, s_col, &one);

  /* D[:, idx] = f - f_prev  (reads old a->f). */
  memcpy(d_col, f, sizeof(aa_float) * a->dim);
  BLAS(axpy)(&bdim, &neg_onef, a->f, &one, d_col, &one);

  /* g = x - f  (this iteration's residual; needed for the solve RHS). */
  memcpy(a->g, x, sizeof(aa_float) * a->dim);
  BLAS(axpy)(&bdim, &neg_onef, f, &one, a->g, &one);

  /* Y[:, idx] = g - g_prev  (reads old a->g_prev; single-rounding y). */
  memcpy(y_col, a->g, sizeof(aa_float) * a->dim);
  BLAS(axpy)(&bdim, &neg_onef, a->g_prev, &one, y_col, &one);

  /* State advance for next iter: (x_prev, f_prev, g_prev) <- (x, f, g).
   * Must follow all the reads above. */
  memcpy(a->x, x, sizeof(aa_float) * a->dim);
  memcpy(a->f, f, sizeof(aa_float) * a->dim);
  memcpy(a->g_prev, a->g, sizeof(aa_float) * a->dim);

  /* Relaxation scratch (mirror of x); only present when relaxation != 1.0. */
  if (a->x_work) {
    memcpy(a->x_work, x, sizeof(aa_float) * a->dim);
  }

  /* ||g|| = ||x - f|| (current residual norm, used by the safeguard). */
  a->norm_g = BLAS(nrm2)(&bdim, a->g, &one);

  TIME_TOC
}

/* f = (1-relaxation) * \sum_i a_i x_i + relaxation * \sum_i a_i f_i */
static void relax(aa_float *f, AaWork *a, aa_int len) {
  TIME_TIC
  /* x_work = x initially */
  blas_int bdim = (blas_int)(a->dim), one = 1, blen = (blas_int)len;
  aa_float onef = 1.0, neg_onef = -1.0;
  aa_float one_m_relaxation = 1. - a->relaxation;
  /* x_work = x - S * work */
  BLAS(gemv)
  ("NoTrans", &bdim, &blen, &neg_onef, a->S, &bdim, a->work, &one, &onef,
   a->x_work, &one);
  /* f = relaxation * f */
  BLAS(scal)(&bdim, &a->relaxation, f, &one);
  /* f += (1 - relaxation) * x_work */
  BLAS(axpy)(&bdim, &one_m_relaxation, a->x_work, &one, f, &one);
  TIME_TOC
}

/* Solve the regularized normal equations (A'B + rI) γ = A'g via a QR
 * factorization of the augmented matrix [A; √r I]. This avoids squaring
 * the condition number the way the normal-equations path did — the
 * returned γ is accurate down to machine precision × κ(A_aug), whereas
 * gesv on A'B lost precision at κ(A_aug)². On exit, f holds the AA
 * iterate and the function returns ||γ|| (or a negative sentinel on
 * failure). */
static aa_float solve(aa_float *f, AaWork *a, aa_int len) {
  TIME_TIC
  blas_int info = -1, bdim = (blas_int)(a->dim), one = 1, blen = (blas_int)len;
  /* Leading dim is fixed to dim+mem regardless of len so the buffers
   * allocated in aa_init match the strides used here. Unused trailing
   * rows are kept zero by build_augmented. */
  blas_int aug_rows = bdim + (blas_int)a->mem;
  aa_float onef = 1.0, neg_onef = -1.0, aa_norm;
  aa_float *A_src = a->type1 ? a->S : a->Y;
  aa_float *gamma = a->work; /* len-sized; `work` is MAX(mem,dim), safe here */

  aa_float r = (a->regularization > 0) ? compute_regularization(a, len) : 0.0;
  aa_float sqrt_r = (r > 0) ? sqrt(r) : 0.0;

  /* 1. Build A_aug = [A; √r I_len, zero-padded]; factor in place. */
  build_augmented(a->A_aug, A_src, a->dim, a->mem, len, sqrt_r);
  BLAS(geqrf)(&aug_rows, &blen, a->A_aug, &aug_rows, a->tau,
              a->qr_work, &a->qr_lwork, &info);

  /* 2. c_aug = [g; 0]; overwrite with Q' c_aug — first `len` entries are
   *    (Q' [g;0])_{1..len}, which is what the solve consumes. */
  if (info == 0) {
    memcpy(a->c_aug, a->g, a->dim * sizeof(aa_float));
    memset(&a->c_aug[a->dim], 0, a->mem * sizeof(aa_float));
    BLAS(ormqr)
    ("Left", "Trans", &aug_rows, &one, &blen, a->A_aug, &aug_rows, a->tau,
     a->c_aug, &aug_rows, a->qr_work, &a->qr_lwork, &info);
  }

  /* 3. Finish the solve differently by type. */
  if (info == 0) {
    if (a->type1) {
      /* Type-I: W γ = Q' [g;0], where W = Q' [Y; √r I]. Build B_aug, apply
       * Q', copy the mem×mem top block into W, solve with gesv. */
      aa_int i;
      build_augmented(a->B_aug, a->Y, a->dim, a->mem, len, sqrt_r);
      BLAS(ormqr)
      ("Left", "Trans", &aug_rows, &blen, &blen, a->A_aug, &aug_rows, a->tau,
       a->B_aug, &aug_rows, a->qr_work, &a->qr_lwork, &info);
      if (info == 0) {
        for (i = 0; i < len; ++i) {
          memcpy(&a->W[i * len], &a->B_aug[i * aug_rows],
                 len * sizeof(aa_float));
        }
        memcpy(gamma, a->c_aug, len * sizeof(aa_float));
        BLAS(gesv)(&blen, &one, a->W, &blen, a->ipiv, gamma, &blen, &info);
      }
    } else {
      /* Type-II: B = A, so Q' [B; √rI] = R — already stored in the upper
       * triangle of A_aug. Back-solve R γ = (Q'c)_{1..len}. Guard against
       * a zero on the diagonal, which geqrf will leave if A_aug has a rank
       * deficiency (it silently produces a triangular R with zero rows
       * rather than setting info). */
      aa_int i;
      memcpy(gamma, a->c_aug, len * sizeof(aa_float));
      for (i = 0; i < len; ++i) {
        if (a->A_aug[i * aug_rows + i] == 0.0) {
          info = i + 1;
          break;
        }
      }
      if (info == 0) {
        BLAS(trsv)("Upper", "NoTrans", "NonUnit", &blen, a->A_aug,
                   &aug_rows, gamma, &one);
      }
    }
  }

  aa_norm = (info == 0) ? BLAS(nrm2)(&blen, gamma, &one) : -1.0;
  if (a->verbosity > 1) {
    printf("AA type %i, iter: %i, len %i, info: %i, aa_norm %.2e\n",
           a->type1 ? 1 : 2, (int)a->iter, (int)len, (int)info, aa_norm);
  }

  if (info != 0 || !isfinite(aa_norm) || aa_norm >= a->max_weight_norm) {
    if (a->verbosity > 0) {
      printf("Error in AA type %i, iter: %i, len %i, info: %i, aa_norm %.2e\n",
             a->type1 ? 1 : 2, (int)a->iter, (int)len, (int)info, aa_norm);
    }
    a->success = 0;
    aa_reset(a);
    TIME_TOC
    if (!isfinite(aa_norm)) aa_norm = -1.0;
    return (aa_norm < 0) ? aa_norm : -aa_norm;
  }

  /* f -= D γ */
  BLAS(gemv)
  ("NoTrans", &bdim, &blen, &neg_onef, a->D, &bdim, gamma, &one, &onef, f,
   &one);

  if (a->relaxation != 1.0) {
    relax(f, a, len);
  }

  a->success = 1;
  TIME_TOC
  return aa_norm;
}

/*
 * API functions below this line, see aa.h for descriptions.
 */
AaWork *aa_init(aa_int dim, aa_int mem, aa_int type1, aa_float regularization,
                aa_float relaxation, aa_float safeguard_factor,
                aa_float max_weight_norm, aa_int verbosity) {
  TIME_TIC
  AaWork *a;
  if (dim <= 0 || mem < 0 || regularization < 0 ||
      relaxation < 0 || relaxation > 2 ||
      safeguard_factor < 0 || max_weight_norm <= 0) {
    printf("Invalid AA parameters.\n");
    return (AaWork *)0;
  }
  a = (AaWork *)calloc(1, sizeof(AaWork));
  if (!a) {
    printf("Failed to allocate memory for AA.\n");
    return NULL;
  }
  a->type1 = type1;
  a->iter = 0;
  a->dim = dim;
  a->mem = MIN(mem, dim); /* for rank stability */
  a->regularization = regularization;
  a->relaxation = relaxation;
  a->safeguard_factor = safeguard_factor;
  a->max_weight_norm = max_weight_norm;
  a->success = 0;
  a->verbosity = verbosity;
  if (a->mem <= 0) {
    return a;
  }

  a->x = (aa_float *)calloc(a->dim, sizeof(aa_float));
  a->f = (aa_float *)calloc(a->dim, sizeof(aa_float));
  a->g = (aa_float *)calloc(a->dim, sizeof(aa_float));

  a->g_prev = (aa_float *)calloc(a->dim, sizeof(aa_float));

  a->Y = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));
  a->S = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));
  a->D = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));

  {
    aa_int aug_rows = a->dim + a->mem;
    a->A_aug = (aa_float *)calloc((size_t)aug_rows * a->mem, sizeof(aa_float));
    a->c_aug = (aa_float *)calloc((size_t)aug_rows, sizeof(aa_float));
    a->tau = (aa_float *)calloc(a->mem, sizeof(aa_float));
    /* type-I needs a second augmented buffer and a mem×mem gesv scratch */
    if (type1) {
      a->B_aug = (aa_float *)calloc((size_t)aug_rows * a->mem, sizeof(aa_float));
      a->W = (aa_float *)calloc((size_t)a->mem * a->mem, sizeof(aa_float));
      a->ipiv = (blas_int *)calloc(a->mem, sizeof(blas_int));
    } else {
      a->B_aug = NULL;
      a->W = NULL;
      a->ipiv = NULL;
    }

    /* LAPACK workspace query: ask geqrf and ormqr for their preferred lwork,
     * then take the max. lwork = -1 makes the routine write the optimal
     * size into work[0] without doing any factoring. The optimal size is
     * returned in an aa_float slot; round up with ceil before casting so a
     * value like 255.9999 doesn't truncate to 255 and under-allocate. */
    {
      blas_int b_aug = (blas_int)aug_rows;
      blas_int b_mem = (blas_int)a->mem;
      blas_int b_neg_one = -1;
      blas_int info_q = 0;
      aa_float q_geqrf = 0.0, q_ormqr_c = 0.0, q_ormqr_b = 0.0;
      BLAS(geqrf)(&b_aug, &b_mem, a->A_aug, &b_aug, a->tau,
                  &q_geqrf, &b_neg_one, &info_q);
      {
        blas_int b_one = 1;
        BLAS(ormqr)
        ("Left", "Trans", &b_aug, &b_one, &b_mem, a->A_aug, &b_aug, a->tau,
         a->c_aug, &b_aug, &q_ormqr_c, &b_neg_one, &info_q);
      }
      if (type1) {
        BLAS(ormqr)
        ("Left", "Trans", &b_aug, &b_mem, &b_mem, a->A_aug, &b_aug, a->tau,
         a->B_aug, &b_aug, &q_ormqr_b, &b_neg_one, &info_q);
      }
      {
        aa_float lwork_f = q_geqrf;
        if (q_ormqr_c > lwork_f) lwork_f = q_ormqr_c;
        if (q_ormqr_b > lwork_f) lwork_f = q_ormqr_b;
        /* Floor at mem — some LAPACK builds return modest sizes; keep
         * a sane minimum. calloc of zero is implementation-defined. */
        if (lwork_f < (aa_float)a->mem) lwork_f = (aa_float)a->mem;
        a->qr_lwork = (blas_int)ceil(lwork_f);
        a->qr_work = (aa_float *)calloc((size_t)a->qr_lwork, sizeof(aa_float));
      }
    }
  }

  /* work is a dim-sized scratch for aa_safeguard's x_new - f_new, and also
   * the len-sized home for γ inside solve(); size for the larger of the two. */
  a->work = (aa_float *)calloc(MAX(a->mem, a->dim), sizeof(aa_float));

  if (relaxation != 1.0) {
    a->x_work = (aa_float *)calloc(a->dim, sizeof(aa_float));
  } else {
    a->x_work = NULL;
  }

  /* If any allocation failed, free the partial workspace and bail. aa_finish
   * is safe against NULL members. */
  if (!a->x || !a->f || !a->g || !a->g_prev ||
      !a->Y || !a->S || !a->D ||
      !a->A_aug || !a->c_aug || !a->tau || !a->qr_work ||
      (type1 && (!a->B_aug || !a->W || !a->ipiv)) ||
      !a->work ||
      (relaxation != 1.0 && !a->x_work)) {
    printf("Failed to allocate memory for AA.\n");
    aa_finish(a);
    return (AaWork *)0;
  }
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
    aa_norm = solve(f, a, len);
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
  if (a->mem <= 0) {
    /* degenerate workspace, nothing to safeguard against */
    TIME_TOC
    return 0;
  }
  if (!a->success) {
    /* last AA update was not successful, no need for safeguarding */
    TIME_TOC
    return 0;
  }

  /* reset success indicator in case safeguarding called multiple times */
  a->success = 0;

  /* NB: a->work is used here as a dim-sized scratch, but elsewhere (in solve)
   * only as a len-sized (<=mem) scratch. This is why it is allocated with
   * MAX(mem, dim) in aa_init — do not shrink it to mem. */
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
    free(a->Y);
    free(a->S);
    free(a->D);
    free(a->A_aug);
    free(a->B_aug);
    free(a->c_aug);
    free(a->tau);
    free(a->qr_work);
    free(a->W);
    free(a->ipiv);
    free(a->work);
    if (a->x_work) {
      free(a->x_work);
    }
    free(a);
  }
}

void aa_reset(AaWork *a) {
  /* reset to the same logical state as a freshly calloc'd workspace */
  if (!a) {
    return;
  }
  if (a->verbosity > 0) {
    printf("AA reset.\n");
  }
  a->iter = 0;
  a->success = 0;
  a->norm_g = 0;
  if (a->x) {
    memset(a->x, 0, sizeof(aa_float) * a->dim);
  }
  if (a->f) {
    memset(a->f, 0, sizeof(aa_float) * a->dim);
  }
  if (a->g) {
    memset(a->g, 0, sizeof(aa_float) * a->dim);
  }
  if (a->g_prev) {
    memset(a->g_prev, 0, sizeof(aa_float) * a->dim);
  }
  if (a->Y) {
    memset(a->Y, 0, sizeof(aa_float) * a->dim * a->mem);
  }
  if (a->S) {
    memset(a->S, 0, sizeof(aa_float) * a->dim * a->mem);
  }
  if (a->D) {
    memset(a->D, 0, sizeof(aa_float) * a->dim * a->mem);
  }
  if (a->A_aug) {
    memset(a->A_aug, 0,
           sizeof(aa_float) * (size_t)(a->dim + a->mem) * a->mem);
  }
  if (a->B_aug) {
    memset(a->B_aug, 0,
           sizeof(aa_float) * (size_t)(a->dim + a->mem) * a->mem);
  }
  if (a->c_aug) {
    memset(a->c_aug, 0, sizeof(aa_float) * (size_t)(a->dim + a->mem));
  }
  if (a->tau) {
    memset(a->tau, 0, sizeof(aa_float) * a->mem);
  }
  if (a->qr_work) {
    memset(a->qr_work, 0, sizeof(aa_float) * (size_t)a->qr_lwork);
  }
  if (a->W) {
    memset(a->W, 0, sizeof(aa_float) * a->mem * a->mem);
  }
  if (a->work) {
    memset(a->work, 0, sizeof(aa_float) * MAX(a->mem, a->dim));
  }
  if (a->ipiv) {
    memset(a->ipiv, 0, sizeof(blas_int) * a->mem);
  }
  if (a->x_work) {
    memset(a->x_work, 0, sizeof(aa_float) * a->dim);
  }
}
