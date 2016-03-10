import sys
import kona
from elasticNozzleMDF import ElasticNozzleMDF

solver = ElasticNozzleMDF(20, 61)

optns = {
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

algorithm = kona.algorithms.Verifier
optimizer = kona.Optimizer(solver, algorithm, optns)
optimizer.solve()
