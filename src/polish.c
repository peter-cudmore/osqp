#include "polish.h"
#include "lin_alg.h"
#include "util.h"
#include "auxil.h"
#include "lin_sys.h"
#include "proj.h"
#include "error.h"
#include "qdldl_interface.h"
#ifdef ENABLE_MKL_PARDISO
#include "pardiso_interface.h"
#endif

/**
 * Update the fixed duplicated active-set matrix:
 *   Ared = vstack[Alow, Aupp]
 * Lower rows occupy [0, m), upper rows occupy [m, 2m).
 */
static void form_Ared(OSQPWorkspace *work) {
  c_int j, ptr;
  c_int nnzA = work->data->A->p[work->data->A->n];

  work->pol->n_low = 0;
  work->pol->n_upp = 0;

  for (j = 0; j < work->data->m; j++) {
    if (work->z[j] - work->data->l[j] < -work->y[j]) {
      work->pol->Alow_to_A[work->pol->n_low++] = j;
      work->pol->A_to_Alow[j] = j;
    } else {
      work->pol->A_to_Alow[j] = -1;
    }

    if (work->data->u[j] - work->z[j] < work->y[j]) {
      work->pol->Aupp_to_A[work->pol->n_upp++] = j;
      work->pol->A_to_Aupp[j] = j;
    } else {
      work->pol->A_to_Aupp[j] = -1;
    }
  }

  vec_set_scalar(work->pol->Ared->x, 0., 2 * nnzA);

  for (ptr = 0; ptr < nnzA; ptr++) {
    j = work->data->A->i[ptr];
    if (work->pol->A_to_Alow[j] != -1) {
      work->pol->Ared->x[work->pol->A_to_Alow_elem[ptr]] = work->data->A->x[ptr];
    }
    if (work->pol->A_to_Aupp[j] != -1) {
      work->pol->Ared->x[work->pol->A_to_Aupp_elem[ptr]] = work->data->A->x[ptr];
    }
  }
}

/**
 * Form fixed-size right-hand side rhs_red = vstack[-q, l_low, u_upp].
 * Inactive rows are left as zero.
 */
static void form_rhs_red(OSQPWorkspace *work, c_float *rhs) {
  c_int j;

  for (j = 0; j < work->data->n; j++) {
    rhs[j] = -work->data->q[j];
  }

  vec_set_scalar(rhs + work->data->n, 0., 2 * work->data->m);

  for (j = 0; j < work->data->m; j++) {
    if (work->pol->A_to_Alow[j] != -1) {
      rhs[work->data->n + j] = work->data->l[j];
    }
    if (work->pol->A_to_Aupp[j] != -1) {
      rhs[work->data->n + work->data->m + j] = work->data->u[j];
    }
  }
}

static c_int update_solver_sigma(OSQPWorkspace *work, c_float sigma) {
  switch (work->settings->linsys_solver) {
  case QDLDL_SOLVER:
    ((qdldl_solver *)work->pol->linsys_solver)->sigma = sigma;
    return 0;
#ifdef ENABLE_MKL_PARDISO
  case MKL_PARDISO_SOLVER:
    ((pardiso_solver *)work->pol->linsys_solver)->sigma = sigma;
    return 0;
#endif
  default:
    return 1;
  }
}

static c_int copy_linsys_solution(OSQPWorkspace *work, c_float *dst) {
  c_int n = work->data->n + 2 * work->data->m;
  switch (work->settings->linsys_solver) {
  case QDLDL_SOLVER:
    prea_vec_copy(((qdldl_solver *)work->pol->linsys_solver)->sol, dst, n);
    return 0;
#ifdef ENABLE_MKL_PARDISO
  case MKL_PARDISO_SOLVER:
    prea_vec_copy(((pardiso_solver *)work->pol->linsys_solver)->sol, dst, n);
    return 0;
#endif
  default:
    return 1;
  }
}

static c_int update_polish_solver(OSQPWorkspace *work) {
  c_int exitflag;

  if (!work->pol->linsys_solver->update_matrices ||
      !work->pol->linsys_solver->update_rho_vec) {
    return 1;
  }

  if (work->pol->delta != work->settings->delta) {
    work->pol->delta = work->settings->delta;
    vec_set_scalar(work->pol->rho_vec, 1. / work->pol->delta, 2 * work->data->m);

    exitflag = update_solver_sigma(work, work->pol->delta);
    if (exitflag) return exitflag;

    exitflag = work->pol->linsys_solver->update_rho_vec(work->pol->linsys_solver,
                                                        work->pol->rho_vec);
    if (exitflag) return exitflag;
  }

  return work->pol->linsys_solver->update_matrices(work->pol->linsys_solver,
                                                   work->data->P,
                                                   work->pol->Ared);
}

/**
 * Perform iterative refinement on the polished solution:
 *    (repeat)
 *    1. (K + dK) * dz = b - K*z
 *    2. z <- z + dz
 */
static c_int iterative_refinement(OSQPWorkspace *work,
                                  LinSysSolver  *p,
                                  c_float       *z,
                                  c_float       *b) {
  c_int i, j, n;
  c_float *rhs;

  if (work->settings->polish_refine_iter <= 0) return 0;

  n   = work->data->n + 2 * work->data->m;
  rhs = work->pol->rhs;

  for (i = 0; i < work->settings->polish_refine_iter; i++) {
    prea_vec_copy(b, rhs, n);

    mat_vec(work->data->P, z, rhs, -1);
    mat_tpose_vec(work->data->P, z, rhs, -1, 1);
    mat_tpose_vec(work->pol->Ared, z + work->data->n, rhs, -1, 0);
    mat_vec(work->pol->Ared, z, rhs + work->data->n, -1);

    if (p->solve(p, rhs) || copy_linsys_solution(work, rhs)) return 1;

    for (j = 0; j < n; j++) {
      z[j] += rhs[j];
    }
  }

  return 0;
}

/**
 * Compute dual variable y from fixed-size yred = vstack[ylow, yupp].
 */
static void get_ypol_from_yred(OSQPWorkspace *work, const c_float *yred) {
  c_int j;

  for (j = 0; j < work->data->m; j++) {
    work->pol->y[j] = 0.0;

    if (work->pol->A_to_Alow[j] != -1) {
      work->pol->y[j] += yred[j];
    }
    if (work->pol->A_to_Aupp[j] != -1) {
      work->pol->y[j] += yred[work->data->m + j];
    }
  }
}

c_int polish(OSQPWorkspace *work) {
  c_int exitflag;
  c_int polish_successful;
  LinSysSolver *plsh;

#ifdef PROFILING
  osqp_tic(work->timer);
#endif /* ifdef PROFILING */

  plsh = work->pol->linsys_solver;

  form_Ared(work);
  form_rhs_red(work, work->pol->rhs_red);
  prea_vec_copy(work->pol->rhs_red, work->pol->pol_sol,
                work->data->n + 2 * work->data->m);

  exitflag = update_polish_solver(work);
  if (exitflag) {
    work->info->status_polish = -1;
    return 1;
  }

  exitflag = plsh->solve(plsh, work->pol->pol_sol);
  if (exitflag || copy_linsys_solution(work, work->pol->pol_sol)) {
    work->info->status_polish = -1;
    return 1;
  }

  exitflag = iterative_refinement(work, plsh, work->pol->pol_sol, work->pol->rhs_red);
  if (exitflag) {
    work->info->status_polish = -1;
    return -1;
  }

  prea_vec_copy(work->pol->pol_sol, work->pol->x, work->data->n);
  mat_vec(work->data->A, work->pol->x, work->pol->z, 0);
  get_ypol_from_yred(work, work->pol->pol_sol + work->data->n);

  project_normalcone(work, work->pol->z, work->pol->y);

  update_info(work, 0, 1, 1);

  polish_successful = (work->pol->pri_res < work->info->pri_res &&
                       work->pol->dua_res < work->info->dua_res) ||
                      (work->pol->pri_res < work->info->pri_res &&
                       work->info->dua_res < 1e-10) ||
                      (work->pol->dua_res < work->info->dua_res &&
                       work->info->pri_res < 1e-10);

  if (polish_successful) {
    work->info->obj_val       = work->pol->obj_val;
    work->info->pri_res       = work->pol->pri_res;
    work->info->dua_res       = work->pol->dua_res;
    work->info->status_polish = 1;

    prea_vec_copy(work->pol->x, work->x, work->data->n);
    prea_vec_copy(work->pol->z, work->z, work->data->m);
    prea_vec_copy(work->pol->y, work->y, work->data->m);

#ifdef PRINTING
    if (work->settings->verbose) print_polish(work);
#endif /* ifdef PRINTING */
  } else {
    work->info->status_polish = -1;
  }

  return 0;
}
