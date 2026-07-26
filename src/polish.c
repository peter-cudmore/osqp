#include "polish.h"
#include "lin_alg.h"
#include "osqp_api_constants.h"
#include "printing.h"
#include "util.h"
#include "auxil.h"
#include "error.h"
#include "timing.h"

#ifdef OSQP_ALGEBRA_BUILTIN
#include "qdldl_interface.h"
#endif
#ifdef OSQP_ALGEBRA_MKL
#include "pardiso_interface.h"
#endif

static void update_active_flags(OSQPWorkspace* work) {
  OSQPInt j;
  OSQPFloat* z = OSQPVectorf_data(work->z);
  OSQPFloat* y = OSQPVectorf_data(work->y);
  OSQPFloat* l = OSQPVectorf_data(work->data->l);
  OSQPFloat* u = OSQPVectorf_data(work->data->u);
  OSQPInt* active_flags = work->pol->active_flags_i;

  work->pol->n_active = 0;

  for (j = 0; j < work->data->m; j++) {
    if ((z[j] - l[j] < -y[j]) || (l[j] == u[j])) {
      active_flags[j] = -1;
      work->pol->n_active++;
    }
    else if (u[j] - z[j] < y[j]) {
      active_flags[j] = 1;
      work->pol->n_active++;
    }
    else {
      active_flags[j] = 0;
    }
  }

  OSQPVectori_from_raw(work->pol->active_flags, active_flags);
}

static OSQPInt form_Ared_dynamic(OSQPWorkspace* work) {
  update_active_flags(work);
  work->pol->Ared = OSQPMatrix_submatrix_byrows(work->data->A, work->pol->active_flags);
  return work->pol->Ared ? OSQP_NO_ERROR : osqp_error(OSQP_MEM_ALLOC_ERROR);
}

static void form_Ared_direct(OSQPWorkspace* work) {
  OSQPInt j;
  OSQPInt nnzA = OSQPMatrix_get_nz(work->data->A);
  OSQPInt* A_i = OSQPMatrix_get_i(work->data->A);
  OSQPFloat* A_x = OSQPMatrix_get_x(work->data->A);
  OSQPFloat* Ared_x = OSQPMatrix_get_x(work->pol->Ared);
  OSQPInt* active_flags = work->pol->active_flags_i;

  update_active_flags(work);
  for (j = 0; j < 2 * nnzA; j++) Ared_x[j] = 0.0;

  for (j = 0; j < nnzA; j++) {
    if (active_flags[A_i[j]] == -1) {
      Ared_x[work->pol->A_to_Alow_elem[j]] = A_x[j];
    }
    else if (active_flags[A_i[j]] == 1) {
      Ared_x[work->pol->A_to_Aupp_elem[j]] = A_x[j];
    }
  }
}

static OSQPInt form_rhs_red_dynamic(OSQPWorkspace* work, OSQPVectorf* rhs) {
  OSQPInt j, counter;
  OSQPFloat* rhsv = OSQPVectorf_data(rhs);
  OSQPFloat* q = OSQPVectorf_data(work->data->q);
  OSQPFloat* l = OSQPVectorf_data(work->data->l);
  OSQPFloat* u = OSQPVectorf_data(work->data->u);
  OSQPInt* active_flags = work->pol->active_flags_i;

  for (j = 0; j < work->data->n; j++) rhsv[j] = -q[j];

  counter = 0;
  for (j = 0; j < work->data->m; j++) {
    if (active_flags[j] == -1) {
      rhsv[work->data->n + counter] = l[j];
      counter++;
    }
    else if (active_flags[j] == 1) {
      rhsv[work->data->n + counter] = u[j];
      counter++;
    }
  }

  return OSQP_NO_ERROR;
}

static void form_rhs_red_direct(OSQPWorkspace* work, OSQPVectorf* rhs) {
  OSQPInt j;
  OSQPFloat* rhsv = OSQPVectorf_data(rhs);
  OSQPFloat* q = OSQPVectorf_data(work->data->q);
  OSQPFloat* l = OSQPVectorf_data(work->data->l);
  OSQPFloat* u = OSQPVectorf_data(work->data->u);
  OSQPInt* active_flags = work->pol->active_flags_i;

  for (j = 0; j < work->data->n; j++) rhsv[j] = -q[j];
  for (j = 0; j < 2 * work->data->m; j++) rhsv[work->data->n + j] = 0.0;

  for (j = 0; j < work->data->m; j++) {
    if (active_flags[j] == -1) {
      rhsv[work->data->n + j] = l[j];
    }
    else if (active_flags[j] == 1) {
      rhsv[work->data->n + work->data->m + j] = u[j];
    }
  }
}

static OSQPInt update_polish_solver_direct(OSQPSolver* solver) {
  OSQPInt exitflag;
  OSQPWorkspace* work = solver->work;
  LinSysSolver* plsh = work->pol->linsys_solver;

  if (!plsh || !plsh->update_matrices || !plsh->update_rho_vec) {
    return 1;
  }

  plsh->update_settings(plsh, solver->settings);

  if (work->pol->delta != solver->settings->delta) {
    work->pol->delta = solver->settings->delta;
    OSQPVectorf_set_scalar(work->pol->rho_vec, 1. / work->pol->delta);
    exitflag = plsh->update_rho_vec(plsh, work->pol->rho_vec, 1. / work->pol->delta);
    if (exitflag) return exitflag;
  }

  return plsh->update_matrices(plsh,
                               work->data->P, OSQP_NULL, OSQPMatrix_get_nz(work->data->P),
                               work->pol->Ared, OSQP_NULL, OSQPMatrix_get_nz(work->pol->Ared));
}

static OSQPInt copy_direct_linsys_solution(LinSysSolver* plsh, OSQPVectorf* dst) {
#ifdef OSQP_ALGEBRA_BUILTIN
  OSQPVectorf_from_raw(dst, ((qdldl_solver *)plsh)->sol);
  return 0;
#elif defined(OSQP_ALGEBRA_MKL)
  OSQPVectorf_from_raw(dst, ((pardiso_solver *)plsh)->sol);
  return 0;
#else
  (void)plsh;
  (void)dst;
  return 1;
#endif
}

static OSQPInt iterative_refinement_dynamic(OSQPSolver*   solver,
                                            LinSysSolver* p,
                                            OSQPVectorf*  z,
                                            OSQPVectorf*  b) {
  OSQPInt i, mred;
  OSQPVectorf *rhs, *rhs1, *rhs2;
  OSQPVectorf *z1, *z2;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  if (settings->polish_refine_iter > 0) {
    mred = OSQPMatrix_get_m(work->pol->Ared);

    rhs = OSQPVectorf_malloc(work->data->n + mred);
    rhs1 = OSQPVectorf_view(rhs, 0, work->data->n);
    rhs2 = OSQPVectorf_view(rhs, work->data->n, mred);
    z1   = OSQPVectorf_view(z, 0, work->data->n);
    z2   = OSQPVectorf_view(z, work->data->n, mred);

    if (!rhs || !rhs1 || !rhs2 || !z1 || !z2) {
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
    }

    for (i = 0; i < settings->polish_refine_iter; i++) {
      OSQPVectorf_copy(rhs, b);
      OSQPMatrix_Axpy(work->data->P, z1, rhs1, -1.0, 1.0);
      OSQPMatrix_Atxpy(work->pol->Ared, z2, rhs1, -1.0, 1.0);
      OSQPMatrix_Axpy(work->pol->Ared, z1, rhs2, -1.0, 1.0);
      p->solve(p, rhs, 1);
      OSQPVectorf_plus(z, z, rhs);
    }

    OSQPVectorf_free(rhs);
    OSQPVectorf_view_free(rhs1);
    OSQPVectorf_view_free(rhs2);
    OSQPVectorf_view_free(z1);
    OSQPVectorf_view_free(z2);
  }

  return 0;
}

static OSQPInt iterative_refinement_direct(OSQPSolver*   solver,
                                           LinSysSolver* p,
                                           OSQPVectorf*  z,
                                           OSQPVectorf*  b) {
  OSQPInt i;
  OSQPWorkspace* work = solver->work;

  if (solver->settings->polish_refine_iter <= 0) return 0;

  for (i = 0; i < solver->settings->polish_refine_iter; i++) {
    OSQPVectorf_copy(work->pol->rhs, b);
    OSQPMatrix_Axpy(work->data->P, work->pol->pol_sol_xview, work->pol->rhs_xview, -1.0, 1.0);
    OSQPMatrix_Atxpy(work->pol->Ared, work->pol->pol_sol_yview, work->pol->rhs_xview, -1.0, 1.0);
    OSQPMatrix_Axpy(work->pol->Ared, work->pol->pol_sol_xview, work->pol->rhs_yview, -1.0, 1.0);

    if (p->solve(p, work->pol->rhs, 1) || copy_direct_linsys_solution(p, work->pol->rhs)) {
      return 1;
    }

    OSQPVectorf_plus(z, z, work->pol->rhs);
  }

  return 0;
}

static OSQPInt get_ypol_from_yred_dynamic(OSQPWorkspace* work, OSQPVectorf* yred_vf) {
  OSQPInt j, counter;
  OSQPFloat* y = OSQPVectorf_data(work->pol->y);
  OSQPFloat* yred = OSQPVectorf_data(yred_vf);
  OSQPInt* active_flags = work->pol->active_flags_i;

  if (work->pol->n_active == 0) {
    OSQPVectorf_set_scalar(work->pol->y, 0.);
    return OSQP_NO_ERROR;
  }

  counter = 0;
  for (j = 0; j < work->data->m; j++) {
    if (active_flags[j] == 0) {
      y[j] = 0.0;
    }
    else {
      y[j] = yred[counter++];
    }
  }

  return OSQP_NO_ERROR;
}

static void get_ypol_from_yred_direct(OSQPWorkspace* work) {
  OSQPInt j;
  OSQPFloat* y = OSQPVectorf_data(work->pol->y);
  OSQPFloat* yred = OSQPVectorf_data(work->pol->pol_sol_yview);
  OSQPInt* active_flags = work->pol->active_flags_i;

  for (j = 0; j < work->data->m; j++) {
    if (active_flags[j] == -1) y[j] = yred[j];
    else if (active_flags[j] == 1) y[j] = yred[work->data->m + j];
    else y[j] = 0.0;
  }
}

static OSQPInt polish_direct(OSQPSolver* solver) {
  OSQPInt polish_successful = 0;
  OSQPInt exitflag = 0;
  LinSysSolver* plsh = solver->work->pol->linsys_solver;
  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  osqp_tic(work->timer);
#endif

  form_Ared_direct(work);
  if (work->pol->n_active == 0) {
    c_print("Polishing not needed - no active set detected at optimal point\n");
    info->status_polish = OSQP_POLISH_NO_ACTIVE_SET_FOUND;
    return OSQP_NO_ERROR;
  }

  form_rhs_red_direct(work, work->pol->rhs_red);
  OSQPVectorf_copy(work->pol->pol_sol, work->pol->rhs_red);

  exitflag = update_polish_solver_direct(solver);
  if (exitflag) {
    info->status_polish = OSQP_POLISH_LINSYS_ERROR;
    return exitflag;
  }

  plsh->warm_start(plsh, work->x);
  exitflag = plsh->solve(plsh, work->pol->pol_sol, 1);
  if (exitflag || copy_direct_linsys_solution(plsh, work->pol->pol_sol)) {
    info->status_polish = OSQP_POLISH_FAILED;
    return exitflag ? exitflag : 1;
  }

  exitflag = iterative_refinement_direct(solver, plsh, work->pol->pol_sol, work->pol->rhs_red);
  if (exitflag) {
    info->status_polish = OSQP_POLISH_FAILED;
    return exitflag;
  }

  OSQPVectorf_copy(work->pol->x, work->pol->pol_sol_xview);
  OSQPMatrix_Axpy(work->data->A, work->pol->x, work->pol->z, 1.0, 0.0);
  get_ypol_from_yred_direct(work);

  OSQPVectorf_plus(work->pol->y, work->pol->y, work->pol->z);
  OSQPVectorf_ew_bound_vec(work->pol->z, work->pol->y, work->data->l, work->data->u);
  OSQPVectorf_minus(work->pol->y, work->pol->y, work->pol->z);

  update_info(solver, 0, 1);

  polish_successful = (work->pol->prim_res < info->prim_res &&
                       work->pol->dual_res < info->dual_res) ||
                      (work->pol->prim_res < info->prim_res &&
                       info->dual_res < 1e-10) ||
                      (work->pol->dual_res < info->dual_res &&
                       info->prim_res < 1e-10);

  if (polish_successful) {
    info->obj_val       = work->pol->obj_val;
    info->dual_obj_val  = work->pol->dual_obj_val;
    info->duality_gap   = work->pol->duality_gap;
    info->prim_res      = work->pol->prim_res;
    info->dual_res      = work->pol->dual_res;
    info->status_polish = OSQP_POLISH_SUCCESS;

    OSQPVectorf_copy(work->x, work->pol->x);
    OSQPVectorf_copy(work->z, work->pol->z);
    OSQPVectorf_copy(work->y, work->pol->y);

#ifdef OSQP_ENABLE_PRINTING
    if (settings->verbose) print_polish(solver);
#endif
  }
  else {
    info->status_polish = OSQP_POLISH_FAILED;
  }

  return OSQP_NO_ERROR;
}

OSQPInt polish(OSQPSolver* solver) {
  OSQPInt polish_successful = 0;
  OSQPInt exitflag = 0;

  LinSysSolver* plsh = OSQP_NULL;
  OSQPVectorf*  rhs_red = OSQP_NULL;
  OSQPVectorf*  pol_sol = OSQP_NULL;
  OSQPVectorf*  pol_sol_xview = OSQP_NULL;
  OSQPVectorf*  pol_sol_yview = OSQP_NULL;

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  if (work->pol->linsys_solver) {
    return polish_direct(solver);
  }

#ifdef OSQP_ENABLE_PROFILING
  osqp_tic(work->timer);
#endif

  exitflag = form_Ared_dynamic(work);
  if (exitflag) {
    info->status_polish = OSQP_POLISH_FAILED;
    return exitflag;
  }
  else if (work->pol->n_active == 0) {
    c_print("Polishing not needed - no active set detected at optimal point\n");
    info->status_polish = OSQP_POLISH_NO_ACTIVE_SET_FOUND;
    OSQPMatrix_free(work->pol->Ared);
    work->pol->Ared = OSQP_NULL;
    return OSQP_NO_ERROR;
  }

  exitflag = osqp_algebra_init_linsys_solver(&plsh, work->data->P, work->pol->Ared,
                                             OSQP_NULL, settings, OSQP_NULL, OSQP_NULL, 1);
  if (exitflag) {
    info->status_polish = OSQP_POLISH_LINSYS_ERROR;
    OSQPMatrix_free(work->pol->Ared);
    work->pol->Ared = OSQP_NULL;
    return exitflag;
  }

  rhs_red = OSQPVectorf_malloc(work->data->n + work->pol->n_active);
  if (!rhs_red) {
    info->status_polish = OSQP_POLISH_FAILED;
    OSQPMatrix_free(work->pol->Ared);
    work->pol->Ared = OSQP_NULL;
    plsh->free(plsh);
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  exitflag = form_rhs_red_dynamic(work, rhs_red);
  if (exitflag) {
    info->status_polish = OSQP_POLISH_FAILED;
    OSQPMatrix_free(work->pol->Ared);
    work->pol->Ared = OSQP_NULL;
    OSQPVectorf_free(rhs_red);
    plsh->free(plsh);
    return exitflag;
  }

  pol_sol = OSQPVectorf_copy_new(rhs_red);
  if (!pol_sol) {
    info->status_polish = OSQP_POLISH_FAILED;
    OSQPMatrix_free(work->pol->Ared);
    work->pol->Ared = OSQP_NULL;
    OSQPVectorf_free(rhs_red);
    plsh->free(plsh);
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  pol_sol_xview = OSQPVectorf_view(pol_sol, 0, work->data->n);
  pol_sol_yview = OSQPVectorf_view(pol_sol, work->data->n, work->pol->n_active);
  if (!pol_sol_xview || !pol_sol_yview) {
    info->status_polish = OSQP_POLISH_FAILED;
    OSQPMatrix_free(work->pol->Ared);
    work->pol->Ared = OSQP_NULL;
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);
    OSQPVectorf_view_free(pol_sol_xview);
    OSQPVectorf_view_free(pol_sol_yview);
    plsh->free(plsh);
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  plsh->warm_start(plsh, work->x);
  plsh->solve(plsh, pol_sol, 1);

  exitflag = iterative_refinement_dynamic(solver, plsh, pol_sol, rhs_red);
  if (exitflag) {
    info->status_polish = OSQP_POLISH_FAILED;
    OSQPMatrix_free(work->pol->Ared);
    work->pol->Ared = OSQP_NULL;
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);
    OSQPVectorf_view_free(pol_sol_xview);
    OSQPVectorf_view_free(pol_sol_yview);
    plsh->free(plsh);
    return exitflag;
  }

  OSQPVectorf_copy(work->pol->x, pol_sol_xview);
  OSQPMatrix_Axpy(work->data->A, work->pol->x, work->pol->z, 1.0, 0.0);
  get_ypol_from_yred_dynamic(work, pol_sol_yview);

  OSQPVectorf_plus(work->pol->y, work->pol->y, work->pol->z);
  OSQPVectorf_ew_bound_vec(work->pol->z, work->pol->y, work->data->l, work->data->u);
  OSQPVectorf_minus(work->pol->y, work->pol->y, work->pol->z);

  update_info(solver, 0, 1);

  polish_successful = (work->pol->prim_res < info->prim_res &&
                       work->pol->dual_res < info->dual_res) ||
                      (work->pol->prim_res < info->prim_res &&
                       info->dual_res < 1e-10) ||
                      (work->pol->dual_res < info->dual_res &&
                       info->prim_res < 1e-10);

  if (polish_successful) {
    info->obj_val       = work->pol->obj_val;
    info->dual_obj_val  = work->pol->dual_obj_val;
    info->duality_gap   = work->pol->duality_gap;
    info->prim_res      = work->pol->prim_res;
    info->dual_res      = work->pol->dual_res;
    info->status_polish = OSQP_POLISH_SUCCESS;

    OSQPVectorf_copy(work->x, work->pol->x);
    OSQPVectorf_copy(work->z, work->pol->z);
    OSQPVectorf_copy(work->y, work->pol->y);

#ifdef OSQP_ENABLE_PRINTING
    if (settings->verbose) print_polish(solver);
#endif
  }
  else {
    info->status_polish = OSQP_POLISH_FAILED;
  }

  plsh->free(plsh);
  OSQPMatrix_free(work->pol->Ared);
  work->pol->Ared = OSQP_NULL;
  OSQPVectorf_free(rhs_red);
  OSQPVectorf_free(pol_sol);
  OSQPVectorf_view_free(pol_sol_xview);
  OSQPVectorf_view_free(pol_sol_yview);

  return OSQP_NO_ERROR;
}
