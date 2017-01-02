import sys
import numpy as np
import kona
from elasticNozzleIDF import ElasticNozzleIDF

solver = ElasticNozzleIDF(20, 121)

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
    'max_iter' : 100,
    'opt_tol' : 1e-5,
    'feas_tol' : 1e-5,
    'matrix_explicit' : True,
    'globalization' : 'filter',

    'trust' : {
        'init_radius' : 0.5,
        'max_radius' : 4.0,
        'min_radius' : 1e-4,
    },

    'penalty' : {
        'mu_init' : 0.01,
        'mu_pow' : 1.0,
        'mu_max' : 1e6,
    },

    'rsnk' : {
        'precond'       : 'idf_schur',
        'krylov_file'   : 'kona_krylov.dat',
        'subspace_size' : 10,
        'check_res'     : False,
        'rel_tol'       : 0.5,
        'product_tol'   : 1e-6,
    },
}

# verifier = kona.algorithms.Verifier
# optimizer = kona.Optimizer(solver, verifier, verifier_optns)

algorithm = kona.algorithms.FLECS_RSNK
optimizer = kona.Optimizer(solver, algorithm, opt_optns)

optimizer.solve(print_opts=True)
