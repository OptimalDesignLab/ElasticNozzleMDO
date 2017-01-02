import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyublas
import aerostruct_mdf as mda
from kona.user import BaseVector, UserSolver
from plot_nozzle import plot_nozzle

class ENVectorState(BaseVector):
    def __init__(self, idx):
        if idx < 0:
            raise ValueError('Invalid vector index!')
        self.idx = idx

    def get_data(self):
        return mda.get_state_data(self.idx)

    def set_data(self, data):
        mda.set_state_data(self.idx, data)

    def plus(self, vector):
        mda.axpby_s(self.idx, 1.0, self.idx, 1.0, vector.idx)

    def times_scalar(self, value):
        mda.axpby_s(self.idx, value, self.idx, 0.0, -1)

    def times_vector(self, vector):
        mda.times_vector_s(self.idx, vector.idx)

    def equals_value(self, value):
        mda.axpby_s(self.idx, value, -1, 0.0, -1)

    def equals_vector(self, vector):
        mda.axpby_s(self.idx, 1.0, vector.idx, 0.0, -1)

    def equals_ax_p_by(self, a, x, b, y):
        mda.axpby_s(self.idx, a, x.idx, b, y.idx)

    def inner(self, vector):
        return mda.inner_s(self.idx, vector.idx)

    @property
    def infty(self):
        return np.nan

    def exp(self, vector):
        self.equals_vector(vector)
        mda.exp_s(self.idx)

    def log(self, vector):
        self.equals_vector(vector)
        mda.log_s(self.idx)

    def pow(self, power):
        mda.pow_s(self.idx, power)

class ElasticNozzleMDF(UserSolver):

    def __init__(self, ndv, nodes):
        super(ElasticNozzleMDF, self).__init__(
            ndv, num_state = 6*nodes,
            num_eq = 0, num_ineq = 0)
        mda.init_mda(ndv, nodes)
        mda.alloc_design(2)

    def allocate_state(self, num_vecs):
        mda.alloc_state(num_vecs)
        out = []
        for i in xrange(num_vecs):
            out.append(ENVectorState(i))
        return out

    def eval_obj(self, at_design, at_state):
        mda.set_design_data(0, at_design)
        return mda.eval_f(0, at_state.idx)

    def eval_residual(self, at_design, at_state, store_here):
        mda.set_design_data(0, at_design)
        mda.eval_r(0, at_state.idx, store_here.idx)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        mda.set_design_data(0, at_design)
        mda.set_design_data(1, in_vec)
        mda.mult_drdx(0, at_state.idx, 1, out_vec.idx)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        mda.set_design_data(0, at_design)
        mda.mult_drdw(0, at_state.idx, in_vec.idx, out_vec.idx)

    def multiply_dRdX_T(self, at_design, at_state, in_vec):
        mda.set_design_data(0, at_design)
        mda.mult_drdx_t(0, at_state.idx, in_vec.idx, 1)
        return mda.get_design_data(1)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        mda.set_design_data(0, at_design)
        mda.mult_drdw_t(0, at_state.idx, in_vec.idx, out_vec.idx)

    def factor_linear_system(self, at_design, at_state):
        mda.set_design_data(0, at_design)
        mda.factor_precond(0, at_state.idx)

    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        mda.set_design_data(0, at_design)
        mda.factor_precond(0, at_state.idx)
        return mda.apply_precond(0, at_state.idx, in_vec.idx, out_vec.idx)

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        mda.set_design_data(0, at_design)
        mda.factor_precond(0, at_state.idx)
        return mda.apply_precond_t(0, at_state.idx, in_vec.idx, out_vec.idx)

    def eval_dFdX(self, at_design, at_state):
        mda.set_design_data(0, at_design)
        mda.eval_dfdx(0, at_state.idx, 1)
        return mda.get_design_data(1)

    def eval_dFdU(self, at_design, at_state, store_here):
        mda.set_design_data(0, at_design)
        mda.eval_dfdw(0, at_state.idx, store_here.idx)

    def init_design(self):
        mda.init_design(1)
        return mda.get_design_data(1)

    def solve_nonlinear(self, at_design, result):
        mda.set_design_data(0, at_design)
        return mda.solve_nonlinear(0, result.idx)

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        mda.set_design_data(0, at_design)
        return mda.solve_linear(
            0, at_state.idx, rhs_vec.idx, result.idx, rel_tol)

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        mda.set_design_data(0, at_design)
        return mda.solve_adjoint(
            0, at_state.idx, rhs_vec.idx, result.idx, rel_tol)

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):
        mda.set_design_data(1, curr_design)
        mda.info_dump(1, curr_state.idx, curr_adj.idx, num_iter)
        plot_nozzle('nozzle_inner_%i.png'%num_iter)
