import sys
import kona
import numpy as np
from elasticNozzleMDF import ElasticNozzleMDF

solver = ElasticNozzleMDF(20, 121)

verifier_optns = {
    'matrix_explicit'   : True,
    'verify' : {
        'primal_vec'    : True,
        'state_vec'     : True,
        'dual_vec'      : False,
        'gradients'     : True,
        'pde_jac'       : True,
        'cnstr_jac'     : False,
        'red_grad'      : True,
        'lin_solve'     : True,
        'out_file'      : sys.stdout,
    },
}

opt_optns = {
    'info_file' : sys.stdout,
    'max_iter' : 100,
    'opt_tol' : 1e-5,
    'matrix_explicit' : True,
    'globalization' : 'trust',

    'trust' : {
        'init_radius' : 0.5,
        'max_radius' : 4.0,
        'min_radius' : 1e-4,
    },

    'rsnk' : {
        'precond'       : None,
        'krylov_file'   : 'kona_krylov.dat',
        'subspace_size' : 10,
        'check_res'     : False,
        'rel_tol'       : 0.5,
    },
}

# verifier = kona.algorithms.Verifier
# optimizer = kona.Optimizer(solver, verifier, verifier_optns)

algorithm = kona.algorithms.STCG_RSNK
optimizer = kona.Optimizer(solver, algorithm, opt_optns)

optimizer.solve(print_opts=True)
