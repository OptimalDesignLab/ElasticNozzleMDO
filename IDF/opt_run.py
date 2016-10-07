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
        'dual_vec'      : True,
        'gradients'     : True,
        'pde_jac'       : True,
        'cnstr_jac'     : True,
        'red_grad'      : True,
        'lin_solve'     : True,
        'out_file'      : sys.stdout,
    },
}

opt_optns = {
    'info_file' : sys.stdout,
    'max_iter' : 50,
    'opt_tol' : 1e-5,
    'feas_tol' : 1e-5,
    'matrix_explicit' : True,

    'homotopy' : {
                'inner_tol' : 1e-2,
                'inner_maxiter' : 20,
                'nominal_dist' : 1.0,
                'nominal_angle' : 15.0*np.pi/180.,
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
        'check_res'     : False,
        'rel_tol'       : 1e-2,
    },
}

# verifier = kona.algorithms.Verifier
# optimizer = kona.Optimizer(solver, verifier, verifier_optns)

algorithm = kona.algorithms.PredictorCorrectorCnstr
optimizer = kona.Optimizer(solver, algorithm, opt_optns)

optimizer.solve()
