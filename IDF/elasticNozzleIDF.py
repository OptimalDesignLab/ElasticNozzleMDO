import numpy as np
import aerostruct_idf as mda
from kona.user import BaseVector, BaseAllocator, UserSolverIDF

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

class ENVectorDual(BaseVector):
    def __init__(self, idx):
        if idx < 0:
            raise ValueError('Invalid vector index!')
        self.idx = idx

    def plus(self, vector):
        mda.axpby_c(self.idx, 1.0, self.idx, 1.0, vector.idx)

    def times_scalar(self, value):
        mda.axpby_c(self.idx, value, self.idx, 0.0, -1)

    def times_vector(self, vector):
        mda.times_vector_c(self.idx, vector.idx)

    def equals_value(self, value):
        mda.axpby_c(self.idx, value, -1, 0.0, -1)

    def equals_vector(self, vector):
        mda.axpby_c(self.idx, 1.0, vector.idx, 0.0, -1)

    def equals_ax_p_by(self, a, x, b, y):
        mda.axpby_c(self.idx, a, x.idx, b, y.idx)

    def inner(self, vector):
        return mda.inner_c(self.idx, vector.idx)

    @property
    def infty(self):
        return np.nan

    def exp(self, vector):
        self.equals_vector(vector)
        mda.exp_c(self.idx)

    def log(self, vector):
        self.equals_vector(vector)
        mda.log_c(self.idx)

    def pow(self, power):
        mda.pow_c(self.idx, power)

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

    def alloc_dual(self, count):
        mda.alloc_dual(count)
        out = []
        for i in xrange(count):
            out.append(ENVectorDual(i))
        return out

class ElasticNozzleIDF(UserSolverIDF):

    def __init__(self, ndv, nodes, rand_init=False):
        mda.init_mda(ndv, nodes, rand_init)
        self.num_real_design = ndv
        self.num_real_cnstr = 0
        num_dis_state = 3*nodes
        num_state = 2*num_dis_state
        self.allocator = ENAllocator(
            self.num_real_design + 2*nodes,
            num_state,
            self.num_real_cnstr + 2*nodes)

    def eval_obj(self, at_design, at_state):
        return mda.eval_f(at_design.idx, at_state.idx)

    def eval_residual(self, at_design, at_state, store_here):
        mda.eval_r(at_design.idx, at_state.idx, store_here.idx)

    def eval_constraints(self, at_design, at_state, store_here):
        mda.eval_c(at_design.idx, at_state.idx, store_here.idx)

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
        return mda.apply_precond(
            at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        return mda.apply_precond_t(
            at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def multiply_dCdX(self, at_design, at_state, in_vec, out_vec):
        mda.mult_dcdx(at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def multiply_dCdU(self, at_design, at_state, in_vec, out_vec):
        mda.mult_dcdw(at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        mda.mult_dcdx_t(at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

    def multiply_dCdU_T(self, at_design, at_state, in_vec, out_vec):
        mda.mult_dcdw_t(at_design.idx, at_state.idx, in_vec.idx, out_vec.idx)

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

    def restrict_design(self, opType, target):
        if opType == 0:
            mda.zero_targ_state(target.idx)
        elif opType == 1:
            mda.zero_real_design(target.idx)
        else:
            raise ValueError('Unexpected type in restrict_design()!')

    def copy_dual_to_targstate(self, take_from, copy_to):
        mda.copy_dual_to_targ_state(take_from.idx, copy_to.idx)

    def copy_targstate_to_dual(self, take_from, copy_to):
        mda.copy_targ_state_to_dual(take_from.idx, copy_to.idx)
