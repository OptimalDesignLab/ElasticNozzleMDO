import numpy as np
import aerostructMDF as mda
from kona.user import BaseVector, BaseAllocator, UserSolver

class ENVectorDesign(BaseVector):
    def __init__(self, idx):
        if idx < 0:
            raise ValueError('Invalid vector index!')
        self.idx = idx

    def plus(self, vector):
        mda.axpby_d(self.idx, 1.0, self.idx, 1.0, vector.idx)

    def times_scalar(self, value):
        mda.axpby_d(self.idx, value, self.idx, 0.0, -1)

    def times_vector(self, vector):
        mda.times_vector_d(self.idx, vector.idx)

    def equals_value(self, value):
        mda.axpby_d(self.idx, value, -1, 0.0, -1)

    def equals_vector(self, vector):
        mda.axpby_d(self.idx, 1.0, vector.idx, 0.0, -1)

    def equals_ax_p_by(self, a, x, b, y):
        mda.axpby_d(self.idx, a, x.idx, b, y.idx)

    def inner(self, vector):
        return mda.inner_d(self.idx, vector.idx)

    @property
    def infty(self):
        return np.nan

    def exp(self, vector):
        self.equals_vector(vector)
        mda.exp_d(self.idx)

    def log(self, vector):
        self.equals_vector(vector)
        mda.log_d(self.idx)

    def pow(self, power):
        mda.pow_d(self.idx, power)

class ENVectorState(BaseVector):
    def __init__(self, idx):
        if idx < 0:
            raise ValueError('Invalid vector index!')
        self.idx = idx

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

class ENAllocator(BaseAllocator):

    def alloc_primal(self, count):
        mda.alloc_design(count)
        out = []
        for i in xrange(count):
            out.append(ENVectorDesign(i))
        return out

    def alloc_state(self, count):
        mda.alloc_state(count)
        out = []
        for i in xrange(count):
            out.append(ENVectorState(i))
        return out

class ElasticNozzleMDF(UserSolver):

    def __init__(self, ndv, nodes):
        mda.init_mda(ndv, nodes)
        self.allocator = ENAllocator(ndv, 6*nodes, 0)

    def eval_obj(self, at_design, at_state):
        return mda.eval_f(at_design.idx, at_state.idx)

    def eval_residual(self, at_design, at_state, store_here):
        mda.eval_r(at_design.idx, at_state.idx, store_here.idx)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        mda.mult_drdx(at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        mda.mult_drdw(at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        mda.mult_drdx_t(at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        mda.mult_drdw_t(at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def factor_linear_system(self, at_design, at_state):
        mda.factor_precond(at_design.idx, at_state.idx)

    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        mda.factor_precond(at_design.idx, at_state.idx)
        return mda.apply_precond(
            at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        mda.factor_precond(at_design.idx, at_state.idx)
        return mda.apply_precond_t(
            at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        mda.axpby_d(out_vec.idx, 0.0, out_vec.idx, 0.0, -1)

    def multiply_dCdU_T(self, at_design, at_state, in_vec, out_vec):
        mda.axpby_s(out_vec.idx, 0.0, out_vec.idx, 0.0, -1)

    def eval_dFdX(self, at_design, at_state, store_here):
        mda.eval_dfdx(at_design.idx, at_state.idx, store_here.idx)

    def eval_dFdU(self, at_design, at_state, store_here):
        mda.eval_dfdw(at_design.idx, at_state.idx, store_here.idx)

    def init_design(self, store_here):
        mda.init_design(store_here.idx)

    def solve_nonlinear(self, at_design, result):
        return mda.solve_nonlinear(at_design.idx, result.idx)

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        return mda.solve_linear(
            at_design.idx, at_state.idx, rhs_vec.idx, result.idx, rel_tol)

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        return mda.solve_adjoint(
            at_design.idx, at_state.idx, rhs_vec.idx, result.idx, rel_tol)

    def current_solution(self, curr_design, curr_state, curr_adj,
                         curr_dual, num_iter, curr_slack):
        mda.info_dump(curr_design.idx, curr_state.idx, curr_adj.idx, num_iter)
