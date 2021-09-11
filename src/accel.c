#include "scs.h"
#include "accel.h"
#include "linalg.h"
#include "scs_blas.h"
#include "util.h"

/* This file uses Anderson acceleration to improve the convergence of the
 * ADMM iteration z^+ = \phi(z). At each iteration we need to solve a (small)
 * linear system, we do this using LAPACK, first forming the normal equations
 * and using ?posv (fastest, but bad numerical stability), if that fails we
 * switch to using ?gels, which uses a QR factorization (slower, but better
 * numerically). If this fails then we just don't do any acceleration this
 * iteration, however we could fall back further to ?gelsy or other more
 * robust methods if we wanted to.
 */

struct SCS_ACCEL_WORK {
#ifdef LAPACK_LIB_FOUND
  aa_float *d_f;
  aa_float *d_g;
  aa_float *f;
  aa_float *g;
  aa_float *sol;
  aa_float *scratch;
  aa_float *Q;
  aa_float *R;
  aa_float *delta;
  aa_float *dummy_row;
  aa_int k, l;
#endif
  aa_float total_accel_time;
};

#ifdef LAPACK_LIB_FOUND
void BLAS(gemv)(const char *trans, const blas_int *m, const blas_int *n,
                const aa_float *alpha, const aa_float *a, const blas_int *lda,
                const aa_float *x, const blas_int *incx, const aa_float *beta,
                aa_float *y, const blas_int *incy);
void BLAS(geqrf)(blas_int *m, blas_int *n, aa_float *a, blas_int *lda,
                 aa_float *tau, aa_float *work, blas_int *lwork,
                 blas_int *info);
void BLAS(orgqr)(blas_int *m, blas_int *n, blas_int *k, aa_float *a, blas_int *lda,
                 aa_float *tau, aa_float *work, blas_int *lwork,
                 blas_int *info);
void BLAS(trsv)(const char *uplo, const char *trans, const char *diag,
                blas_int *n, aa_float *a, blas_int *lda, aa_float *x,
                blas_int *incx);
void BLAS(rotg)(aa_float *a, aa_float *b, aa_float *c, aa_float *s);
void BLAS(rot)(const blas_int *n, aa_float *x, const blas_int *incx,
               aa_float *y, const blas_int *incy, const aa_float *c,
               const aa_float *s);
aa_float BLAS(nrm2)(const blas_int *n, aa_float *x, const blas_int *incx);

/* d_f * sol = QR * sol = f */
aa_float *solve_accel_linsys(ScsAccelWork *a, aa_int len) {
  DEBUG_FUNC
  blas_int twol = 2 * a->l;
  blas_int one = 1;
  blas_int bk = (blas_int)a->k;
  aa_float onef = 1.0;
  aa_float zerof = 0.0;
  aa_float neg_onef = -1.0;
  /* sol = f */
  memcpy(a->sol, a->f, sizeof(aa_float) * 2 * a->l);
  /* scratch = Q' * sol = R^-T * d_f' * sol*/
  BLAS(gemv)
  ("Trans", &twol, &bk, &onef, a->d_f, &twol, a->sol, &one, &zerof, a->scratch,
   &one);
  BLAS(trsv)
  ("Upper", "Trans", "not_unit_diag", &bk, a->R, &bk, a->scratch, &one);
  /* scratch = R^-1 * scratch */
  BLAS(trsv)
  ("Upper", "no_trans", "not_unit_diag", &bk, a->R, &bk, a->scratch, &one);
  /* sol = g */
  memcpy(a->sol, a->g, sizeof(aa_float) * 2 * a->l);
  /* sol = sol - d_g * scratch */
  BLAS(gemv)
  ("no_trans", &twol, &bk, &neg_onef, a->d_g, &twol, a->scratch, &one, &onef,
   a->sol, &one);
  RETURN a->sol;
}

void update_accel_params(ScsWork *w, aa_int idx) {
  DEBUG_FUNC
  aa_float *d_f = w->accel->d_f;
  aa_float *d_g = w->accel->d_g;
  aa_float *f = w->accel->f;
  aa_float *g = w->accel->g;
  aa_float *delta = w->accel->delta;
  aa_int l = w->m + w->n + 1;
  /* copy old col into delta */
  memcpy(delta, &(d_f[idx * dim]), sizeof(aa_float) * dim);
  /* copy g_prev into idx col of d_g */
  memcpy(&(d_g[idx * dim]), g, sizeof(aa_float) * dim);
  /* copy f_prev into idx col of d_f */
  memcpy(&(d_f[idx * dim]), f, sizeof(aa_float) * dim);
  /* g = [u;v] */
  memcpy(g, w->u, sizeof(aa_float) * l);
  memcpy(&(g[l]), w->v, sizeof(aa_float) * l);
  /* calculate f = g - [u_prev, v_prev] */
  memcpy(f, g, sizeof(aa_float) * dim);
  add_scaled_array(f, w->u_prev, l, -1.0);
  add_scaled_array(&(f[l]), w->v_prev, l, -1.0);
  /* idx col of d_g = g_prev - g */
  add_scaled_array(&(d_g[idx * dim]), g, dim, -1);
  /* idx col of d_f = f_prev - f */
  add_scaled_array(&(d_f[idx * dim]), f, dim, -1);
  /* delta = delta - new col */
  add_scaled_array(delta, &(d_f[idx * dim]), dim, -1.0);
  /* delta = new - old */
  scale_array(delta, -1.0, dim);
  RETURN;
}

ScsAccelWork *init_accel(ScsWork *w) {
  DEBUG_FUNC
  ScsAccelWork *a = scs_calloc(1, sizeof(ScsAccelWork));
  if (!a) {
    RETURN SCS_NULL;
  }
  a->l = w->m + w->n + 1;
  /* k = lookback - 1 since we use the difference form
     of anderson acceleration, and so there is one fewer var in lin sys.
     Use MIN to prevent not full rank matrices */
  a->k = MIN(w->n, w->stgs->acceleration_lookback - 1);
  if (a->k <= 0) {
    RETURN a;
  }
  a->d_f = scs_calloc(2 * a->l * a->k, sizeof(aa_float));
  a->d_g = scs_calloc(2 * a->l * a->k, sizeof(aa_float));
  a->f = scs_calloc(2 * a->l, sizeof(aa_float));
  a->g = scs_calloc(2 * a->l, sizeof(aa_float));
  a->Q = scs_calloc(2 * a->l * a->k, sizeof(aa_float));
  a->R = scs_calloc(a->k * a->k, sizeof(aa_float));
  a->dummy_row = scs_calloc(a->k, sizeof(aa_float));
  a->sol = scs_malloc(sizeof(aa_float) * 2 * a->l);
  a->scratch = scs_malloc(sizeof(aa_float) * 2 * a->l);
  a->delta = scs_malloc(sizeof(aa_float) * 2 * a->l);
  a->total_accel_time = 0.0;
  if (!a->d_f || !a->d_g || !a->f || !a->g || !a->scratch || !a->sol ||
      !a->delta || !a->R) {
    free_accel(a);
    a = SCS_NULL;
  }
  RETURN a;
}

void qrfactorize(ScsAccelWork *a) {
  DEBUG_FUNC
  aa_int l = a->l;
  aa_int i;
  blas_int twol = dim;
  blas_int bk = (blas_int)a->k;
  blas_int neg_one = -1;
  blas_int info;
  blas_int lwork;
  aa_float worksize;
  aa_float *work;
  aa_float *tau = scs_malloc(a->k * sizeof(aa_float));
  aa_float *Q = a->Q;
  memcpy(Q, a->d_f, sizeof(aa_float) * a->k * 2 * a->l);
  BLAS(geqrf)(&twol, &bk, Q, &twol, tau, &worksize, &neg_one, &info);
  lwork = (blas_int)worksize;
  work = scs_malloc(lwork * sizeof(aa_float));
  BLAS(geqrf)(&twol, &bk, Q, &twol, tau, work, &lwork, &info);
  scs_free(work);
  for (i = 0; i < a->k; ++i) {
    memcpy(&(a->R[i * a->k]), &(Q[i * a->l * 2]), sizeof(aa_float) * (i + 1));
  }
  BLAS(orgqr)(&twol, &bk, &bk, Q, &twol, tau, &worksize, &neg_one, &info);
  lwork = (blas_int)worksize;
  work = scs_malloc(lwork * sizeof(aa_float));
  BLAS(orgqr)(&twol, &bk, &bk, Q, &twol, tau, work, &lwork, &info);
  scs_free(work);
  scs_free(tau);
  RETURN;
}

void update_factorization(ScsAccelWork *a, aa_int idx) {
  DEBUG_FUNC
  aa_float *Q = a->Q;
  aa_float *R = a->R;
  aa_float *u = a->delta;
  aa_float *w = a->scratch;
  aa_float *dummy_row = a->dummy_row;
  blas_int one = 1;
  blas_int bk = (blas_int)a->k;
  blas_int twol = (blas_int)2 * a->l;
  aa_float zerof = 0.0;
  aa_float onef = 1.0;
  aa_float neg_onef = -1.0;
  aa_float nrm_u;
  aa_float c;
  aa_float s;

  aa_int k = a->k;
  aa_int l = a->l;
  aa_float r1, r2;
  aa_int i, ridx;

  memset(dummy_row, 0, k * sizeof(aa_float));
  /* w = Q' * delta, size k: col of R */
  BLAS(gemv)("Trans", &twol, &bk, &onef, a->Q, &twol, u, &one, &zerof, w, &one);
  /* u = delta - Q * w = d_f * R^-1 w, size m: col of Q */
  BLAS(gemv)("no_trans", &twol, &bk, &neg_onef, a->Q, &twol, w, &one, &onef, u, &one);
  /* nrm_u = ||u|| */
  nrm_u = BLAS(nrm2)(&twol, u, &one);
  /* u = u / ||u|| */
  scale_array(u, 1.0 / nrm_u, dim);
  /* R col += w */
  add_scaled_array(&(R[idx * k]), w, k, 1.0);

  /* Givens rotations, start with fake bottom row of R, extra col of Q */
  ridx = k * idx + k - 1;
  r1 = R[ridx];
  r2 = nrm_u;
  dummy_row[idx] = nrm_u;
  BLAS(rotg)(&r1, &r2, &c, &s);
  BLAS(rot)(&bk, &(R[k - 1]), &bk, dummy_row, &one, &c, &s);
  BLAS(rot)(&twol, &(Q[dim * (k - 1)]), &one, u, &one, &c, &s);

  /* Walk up the spike, R finishes upper Hessenberg */
  for (i = k; i > idx + 1; --i) {
    ridx = k * idx + i - 1;
    r1 = R[ridx - 1];
    r2 = R[ridx];
    BLAS(rotg)(&(r1), &(r2), &c, &s);
    BLAS(rot)(&bk, &(R[i - 2]), &bk, &(R[i - 1]), &bk, &c, &s);
    BLAS(rot)
    (&twol, &(Q[dim * (i - 2)]), &one, &(Q[dim * (i - 1)]), &one, &c, &s);
  }

  /* Walk down the sub-diagonal, R finishes upper triangular */
  for (i = idx + 1; i < k - 1; ++i) {
    ridx = k * i + i;
    r1 = R[ridx];
    r2 = R[ridx + 1];
    BLAS(rotg)(&r1, &r2, &c, &s);
    BLAS(rot)(&bk, &(R[i]), &bk, &(R[i + 1]), &bk, &c, &s);
    BLAS(rot)
    (&twol, &(Q[dim * i]), &one, &(Q[dim * (i + 1)]), &one, &c, &s);
  }

  /* Finish fake bottom row of R, extra col of Q */
  BLAS(rotg)(&(R[k * k - 1]), &(dummy_row[k - 1]), &c, &s);
  BLAS(rot)(&twol, &(Q[dim * (k - 1)]), &one, u, &one, &c, &s);
  RETURN;
}

aa_int accelerate(ScsWork *w, aa_int iter) {
  DEBUG_FUNC
  aa_int l = w->accel->dim;
  aa_int k = w->accel->k;
  aa_float *sol;
  aa_int idx;
  timer accel_timer;
  if (k <= 0) {
    RETURN 0;
  }
  tic(&accel_timer);
  idx = k - 1 - iter % k;
  /* update d_f, d_g, f, g, delta */
  update_accel_params(w, idx);
  /* iter < k doesn't do any acceleration until iters hit k - 1
     at which point idx = 0, the matrix is filled and we can start
     accelerating.
  */
  if (iter < k - 1) {
    RETURN 0;
  }
  if (idx == 0) {
    /* every k iterations this does a full factorization for
       numerical stability
    */
    qrfactorize(w->accel);
  } else {
    /* update Q, R factors */
    update_factorization(w->accel, idx);
  }
  /* solve linear system, new point stored in sol */
  sol = solve_accel_linsys(w->accel, k);
  /* set [u;v] = sol */
  memcpy(w->u, sol, sizeof(aa_float) * l);
  memcpy(w->v, &(sol[l]), sizeof(aa_float) * l);
  w->accel->total_accel_time += tocq(&accel_timer);
  RETURN 0;
}

void free_accel(ScsAccelWork *a) {
  DEBUG_FUNC
  if (a) {
    if (a->d_f) scs_free(a->d_f);
    if (a->d_g) scs_free(a->d_g);
    if (a->f) scs_free(a->f);
    if (a->g) scs_free(a->g);
    if (a->sol) scs_free(a->sol);
    if (a->scratch) scs_free(a->scratch);
    if (a->Q) scs_free(a->Q);
    if (a->R) scs_free(a->R);
    if (a->delta) scs_free(a->delta);
    if (a->dummy_row) scs_free(a->dummy_row);
    scs_free(a);
  }
  RETURN;
}

#else

ScsAccelWork *init_accel(ScsWork *w) {
  ScsAccelWork *a = scs_malloc(sizeof(ScsAccelWork));
  a->total_accel_time = 0.0;
  RETURN a;
}

void free_accel(ScsAccelWork *a) {
  if (a) {
    scs_free(a);
  }
}

aa_int accelerate(ScsWork *w, aa_int iter) { RETURN 0; }
#endif

char *get_accel_summary(const ScsInfo *info, ScsAccelWork *a) {
  DEBUG_FUNC
  char *str = scs_malloc(sizeof(char) * 64);
  sprintf(str, "\tAcceleration: avg step time: %1.2es\n",
          a->total_accel_time / (info->iter + 1) / 1e3);
  a->total_accel_time = 0.0;
  RETURN str;
}
