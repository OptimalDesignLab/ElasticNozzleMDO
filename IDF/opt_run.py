import sys
import numpy as np
import kona
from elasticNozzleIDF import ElasticNozzleIDF

solver = ElasticNozzleIDF(20, 61)

verifier_optns = {
    'matrix_explicit'   : True,
    'verify' : {
        'primal_vec'    : True,
        'state_vec'     : True,
        'dual_vec_eq'   : True,
        'gradients'     : True,
        'pde_jac'       : True,
        'cnstr_jac_eq'  : True,
        'red_grad'      : True,
        'lin_solve'     : True,
        'out_file'      : sys.stdout,
    },
}

opt_optns = {
    'info_file' : sys.stdout,
    'max_iter' : 1,
    'opt_tol' : 1e-5,
    'feas_tol' : 1e-5,
    'globalization' : None,
    'matrix_explicit' : True,

    'homotopy' : {
        'inner_tol' : 1e-2,
        'inner_maxiter' : 50,
        'nominal_dist' : 5.0,
        'nominal_angle' : 15.0*np.pi/180.,
    },

    'trust' : {
        'init_radius' : 1.0,
        'max_radius' : 2.0,
        'min_radius' : 1e-4,
    },

    'penalty' : {
        'mu_init' : 0.01,
        'mu_pow' : 1.0,
        'mu_max' : 1e5,
    },

    'rsnk' : {
        'precond'       : 'idf_schur',
        # rsnk algorithm settings
        'dynamic_tol'   : False,
        'nu'            : 1.0,
        # reduced KKT matrix settings
        'product_fac'   : 300.0,
        'lambda'        : 0.0,
        'scale'         : 1.0,
        'grad_scale'    : 1.0,
        'feas_scale'    : 1.0,
        # krylov solver settings
        'krylov_file'   : 'kona_krylov.dat',
        'subspace_size' : 10,
        'check_res'     : True,
        'rel_tol'       : 1e-2,
    },
}

verifier = kona.algorithms.Verifier
optimizer = kona.Optimizer(solver, verifier, verifier_optns)

# algorithm = kona.algorithms.PredictorCorrectorCnstr
# algorithm = kona.algorithms.FLECS_RSNK
# optimizer = kona.Optimizer(solver, algorithm, opt_optns)

optimizer.solve()
