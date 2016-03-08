/**
 * \file krylov.hpp
 * \brief templated Krylov-subspace methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 * \version 1.0
 */

#pragma once

#include <math.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/tuple/tuple.hpp>
#include <ostream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

namespace kona {

using std::cerr;
using std::cout;
using std::endl;
using std::ostream;
using std::ios;
using std::numeric_limits;
using std::string;
using std::vector;
namespace ublas = boost::numeric::ublas;
using ::boost::property_tree::ptree;
using ::boost::tuples::tuple;
using ::boost::tuples::tie;
using ::boost::tuples::make_tuple;

/// machine epsilon
const double kEpsilon = numeric_limits<double>::epsilon();

// ==============================================================================
/*!
 * \class MatrixVectorProduct
 * \brief abstract base class for defining matrix-vector products
 * \tparam DomainVector - generic vector in the matrix domain space
 * \tparam RangeVector - generic vector in the matrix range space
 *
 * The Krylov-subspace solvers require only matrix-vector products and
 * not the actual matrix/Jacobian.  We need some way to indicate which
 * function will perform the product.  However, sometimes the
 * functions that define the product will require different numbers
 * and types of inputs.  For example, the forward-difference
 * approximation to a Jacobian-vector product requires the vector that
 * defines the Jacobian and a perturbation parameter.  The
 * MatrixVectorProduct class is used to derive child classes that can
 * handle the different types of matrix-vector products and still be
 * passed to a single implementation of the Krylov solvers.
 */
template <class DomainVector, class RangeVector = DomainVector>
class MatrixVectorProduct {
 public:
  virtual ~MatrixVectorProduct() {} ///< class destructor
  virtual void MemoryRequired(ptree & num_required) {} ///< user vectors req.
  virtual void Initialize(const ptree& prod_param) {} ///< post-constructor init
  virtual void set_product_tol(const double & tol) {} ///< dynamic tolerance
  virtual void operator()(const DomainVector & u,
                          RangeVector & v) = 0; ///< matrix-vector product
};
// ==============================================================================
/*!
 * \class Preconditioner
 * \brief abstract base class for defining preconditioning operation
 * \tparam DomainVector - generic vector in the matrix domain space
 * \tparam RangeVector - generic vector in the matrix range space
 *
 * See the remarks regarding the MatrixVectorProduct class.  The same
 * idea applies here to the preconditioning operation.
 */
template <class DomainVector, class RangeVector = DomainVector>
class Preconditioner {
 public:
  virtual ~Preconditioner() {} ///< class destructor
  virtual void MemoryRequired(ptree & num_required) {} ///< user vectors req.
  virtual void set_diagonal(const double & diag) {}; ///< add a diagonal matrix
  virtual void operator()(DomainVector & u,
                          RangeVector & v) = 0; ///< preconditioning operation
};
// ==============================================================================
/*!
 * \class IterativeSolver
 * \brief abstract base class for defining iterative solvers
 * \tparam DomainVector - generic vector in the matrix domain space
 * \tparam RangeVector - generic vector in the matrix range space
 *
 * Hopefully, all Krylov solvers will transition to using this base class.
 */
template <class DomainVector, class RangeVector = DomainVector>
class IterativeSolver {
 public:
  virtual ~IterativeSolver() {} ///< class destructor
  virtual void SubspaceSize(int m) = 0; ///< sets the solution subspace size
  virtual void MemoryRequired(ptree& num_required) const = 0; ///< memory needed
  virtual void Solve(const ptree & ptin, const RangeVector & b, DomainVector & x,
                     MatrixVectorProduct<DomainVector,RangeVector> & mat_vec,
                     Preconditioner<DomainVector,RangeVector> & precond,
                     ptree & ptout, ostream & fout = cout) = 0; ///< solve
  virtual void ReSolve(const ptree & ptin, const RangeVector & b,
                       DomainVector & x, ptree & ptout,
                       ostream & fout) {} ///< resolve using subspace
};
// ==============================================================================
/*!
 * \class FGMRESSolver
 * \brief Flexible Generalized Minimal Residual (FGMRES) Krylov iterative method
 * \tparam Vec - generic vector in the matrix domain and range
 */
template <class Vec>
class FGMRESSolver : public IterativeSolver<Vec,Vec> {
 public:
  /*!
   * \brief default constructor
   */
  FGMRESSolver();

  /*!
   * \brief destructor
   */
  ~FGMRESSolver() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

  /*!
   * \brief indicates how many user vectors are needed
   * \param[out] num_required - stores the number of user vectors required
   *
   * num_required["num_vec"] stores number of Vec vectors required
   */
  void MemoryRequired(ptree& num_required) const;
  
  /*!
   * \brief Solve a linear system using FGMRES
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs
   * \param[in,out] x - the solution; on entry, the initial guess
   * \param[in] mat_vec - object that defines matrix-vector product
   * \param[in] precond - object that defines preconditioner
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void Solve(const ptree & ptin, const Vec & b, Vec & x,
             MatrixVectorProduct<Vec,Vec> & mat_vec,
             Preconditioner<Vec,Vec> & precond,
             ptree & ptout, ostream & fout = cout);

 private:
  int maxiter_; ///< maximum number of Krylov iterations
  std::vector<Vec> W_; ///< vector space that spans the residual norm
  std::vector<Vec> Z_; ///< vector space that the solution is built from
  ublas::vector<double> g_; ///< reduced-space rhs
  ublas::vector<double> y_; ///< reduced-space solution
  ublas::vector<double> sn_; ///< sines for Givens rotations
  ublas::vector<double> cs_; ///< cosines for Givens rotations
  ublas::matrix<double> H_; ///< matrix from Arnoldi process
};
// ==============================================================================
/*!
 * \class FFOMSolver
 * \brief Flexible Full-Orthogonalization Method (FFOM) Krylov iterative method
 * \tparam Vec - generic vector in the matrix domain and range
 */
template <class Vec>
class FFOMSolver : public IterativeSolver<Vec,Vec> {
 public:
  /*!
   * \brief default constructor
   */
  FFOMSolver();

  /*!
   * \brief destructor
   */
  ~FFOMSolver() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

  /*!
   * \brief indicates how many user vectors are needed
   * \param[out] num_required - stores the number of user vectors required
   *
   * num_required["num_vec"] stores number of Vec vectors required
   */
  void MemoryRequired(ptree& num_required) const;
  
  /*!
   * \brief Solve a linear system using FFOM
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs
   * \param[in,out] x - the solution; on entry, the initial guess
   * \param[in] mat_vec - object that defines matrix-vector product
   * \param[in] precond - object that defines preconditioner
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void Solve(const ptree & ptin, const Vec & b, Vec & x,
             MatrixVectorProduct<Vec,Vec> & mat_vec,
             Preconditioner<Vec,Vec> & precond,
             ptree & ptout, ostream & fout = cout);

 private:
  int maxiter_; ///< maximum number of Krylov iterations
  std::vector<Vec> W_; ///< vector space that spans the residual norm
  std::vector<Vec> Z_; ///< vector space that the solution is built from
  ublas::vector<double> g_; ///< reduced-space rhs
  ublas::vector<double> y_; ///< reduced-space solution
  ublas::vector<double> y_old_; ///< previous iteration reduced-space solution
  ublas::vector<double> sn_; ///< sines for Givens rotations
  ublas::vector<double> cs_; ///< cosines for Givens rotations
  ublas::matrix<double> H_; ///< matrix from Arnoldi process
};
// ==============================================================================
/*!
 * \class MINRESSolver
 * \brief Minimum  (MINRES) Krylov iterative method
 * \tparam Vec - generic vector in the matrix domain and range
 *
 * This is an implementation of MINRES (see C. C. Paige and M. A. Saunders
 * (1975), Solution of sparse indefinite systems of linear equations, SIAM
 * J. Numer. Anal. 12(4), pp. 617-629).  The code is based on the F90 version
 * developed by the Stanford Systems Optimization Laboratory.
 */
template <class Vec>
class MINRESSolver : public IterativeSolver<Vec,Vec> {
 public:
  /*!
   * \brief default constructor
   */
  MINRESSolver();

  /*!
   * \brief destructor
   */
  ~MINRESSolver() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

  /*!
   * \brief indicates how many user vectors are needed
   * \param[out] num_required - stores the number of user vectors required
   *
   * num_required["num_vec"] stores number of Vec vectors required
   */
  void MemoryRequired(ptree& num_required) const;
  
  /*!
   * \brief Solve a linear system using MINRES
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs
   * \param[in,out] x - the solution; on entry, the initial guess
   * \param[in] mat_vec - object that defines matrix-vector product
   * \param[in] precond - object that defines preconditioner
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void Solve(const ptree & ptin, const Vec & b, Vec & x,
             MatrixVectorProduct<Vec,Vec> & mat_vec,
             Preconditioner<Vec,Vec> & precond,
             ptree & ptout, ostream & fout = cout);

 private:
  int maxiter_; ///< maximum number of Krylov iterations
  std::vector<Vec> work_; ///< "storage" for the user vectors needed  
};
// ==============================================================================
/*!
 * \class FITRSolver
 * \brief Flexible Iterative Trust Region method
 */
template <class Vec>
class FITRSolver : public IterativeSolver<Vec,Vec> {
 public:

  /*!
   * \brief default constructor
   */
  FITRSolver();
  
  /*!
   * \brief class destructor
   */
  ~FITRSolver() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

  /*!
   * \brief indicates how many user vectors are needed
   * \param[out] num_required - stores the number of user vectors required
   *
   * num_required["num_vec"] stores number of Vec vectors required
   */
  void MemoryRequired(ptree& num_required) const;
  
  /*!
   * \brief solves the trust-region subproblem based on the inputs
   * \param[in] inparam - input parameters for solver
   * \param[in] b - the rhs (-gradient)
   * \param[out] x - the solution (assumed that x = 0.0 initially)
   * \param[in] mat_vec - object that defines matrix-vector product for Vec
   * \param[in] precond - object that defines preconditioner for Vec
   * \param[out] outparam - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void Solve(const ptree & ptin, const Vec & b, Vec & x,
             MatrixVectorProduct<Vec> & mat_vec, Preconditioner<Vec> & precond,
             ptree & ptout, ostream & fout = cout);

  /*!
   * \brief recycles the subspace to resolve at a new radius
   * \param[in] inparam - input parameters for solver
   * \param[in] b - the rhs (-gradient)
   * \param[out] x - the solution (assumed that x = 0.0 initially for this solver)
   * \param[out] outparam - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void ReSolve(const ptree & ptin, const Vec & b, Vec & x, ptree & ptout,
               ostream & fout = cout);
  
 private:
  int maxiter_; ///< maximum number of Krylov iterations
  vector<Vec> V_; ///< vector space that spans the residual norm
  vector<Vec> Z_; ///< vector space that the solution is built from
  boost::scoped_ptr<Vec> r_; ///< residual vector
  ublas::vector<double> g_; ///< reduced-space rhs
  ublas::vector<double> y_; ///< reduced-space solution
  ublas::matrix<double> B_; ///< B = V^T Z
  ublas::matrix<double> H_; ///< matrix from Arnoldi process
};
// ==============================================================================
/*!
 * \class STCGSolver
 * \brief Steihaug-Toint Conjugate Gradient (STCG) Krylov iterative method
 * \tparam Vec - generic vector in the matrix domain and range
 */
template <class Vec>
class STCGSolver : public IterativeSolver<Vec,Vec> {
 public:
  /*!
   * \brief default constructor
   */
  STCGSolver();

  /*!
   * \brief destructor
   */
  ~STCGSolver() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

  /*!
   * \brief indicates how many user vectors are needed
   * \param[out] num_required - stores the number of user vectors required
   *
   * num_required["num_vec"] stores number of Vec vectors required
   */
  void MemoryRequired(ptree& num_required) const;
  
  /*!
   * \brief Solve a linear system using STCG
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs
   * \param[in,out] x - the solution; on entry, the initial guess
   * \param[in] mat_vec - object that defines matrix-vector product
   * \param[in] precond - object that defines preconditioner
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void Solve(const ptree & ptin, const Vec & b, Vec & x,
             MatrixVectorProduct<Vec,Vec> & mat_vec,
             Preconditioner<Vec,Vec> & precond,
             ptree & ptout, ostream & fout = cout);

 private:
  int maxiter_; ///< maximum number of Krylov iterations
  std::vector<Vec> work_; ///< "storage" for the user vectors needed  
};
// ==============================================================================
/*!
 * \class FFOMWithSMART
 * \brief FFOM Krylov iterative method for KKT systems using SMART tests 
 * \tparam Vec - generic vector in the matrix domain and range
 */
template <class Vec, class PrimVec, class DualVec>
class FFOMWithSMART : public IterativeSolver<Vec,Vec> {
 public:
  /*!
   * \brief default constructor
   */
  FFOMWithSMART();

  /*!
   * \brief destructor
   */
  ~FFOMWithSMART() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

  /*!
   * \brief indicates how many user vectors are needed
   * \param[out] num_required - stores number of vectors required
   *
   * num_required["num_vec"] stores number of Vec vectors required
   * num_required["primal"] stores number of PrimVec vectors required
   * num_required["dual"] stores number of DualVec vectors required
   */
  void MemoryRequired(ptree& num_required) const;
  
  /*!
   * \brief Solve a linear system using FFOM
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs
   * \param[in,out] x - the solution; on entry, x.primal() holds the obj grad!
   * \param[in] mat_vec - object that defines matrix-vector product
   * \param[in] precond - object that defines preconditioner
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void Solve(const ptree & ptin, const Vec & b, Vec & x,
             MatrixVectorProduct<Vec,Vec> & mat_vec,
             Preconditioner<Vec,Vec> & precond,
             ptree & ptout, ostream & fout = cout);

 private:
  int maxiter_; ///< maximum number of Krylov iterations
  std::vector<Vec> V_; ///< vector space that spans the residual norm
  std::vector<Vec> Z_; ///< vector space that the solution is built from
  boost::scoped_ptr<Vec> res_; ///< residual, b - Ax
  boost::scoped_ptr<DualVec> Ap_; ///< product of search dir. and cnstr. Jacobian
  ublas::vector<double> g_; ///< reduced-space rhs
  ublas::vector<double> y_; ///< reduced-space solution
  ublas::vector<double> sn_; ///< sines for Givens rotations
  ublas::vector<double> cs_; ///< cosines for Givens rotations
  ublas::matrix<double> B_; ///< Hessenburg matrix from Arnoldi process
  ublas::matrix<double> VtZ_; ///< V^T*Z
  ublas::matrix<double> VtZprim_; ///< (V_primal)^T*(Z_primal)
  ublas::matrix<double> VtZdual_; ///< (V_dual)^T(Z_dual)
  
  /*!
   * \brief lower and upper bounds for normal and tangential parts steps
   * \param[in] p - the primal step
   * \param[in] Ap - the product of the primal step with the Jacobian
   * \param[in] A_norm - estimate of the norm of the Jacobian
   * \param[out] upsilon - upper bound on the tangential part of the step
   * \param[out] nu - lower bound on the normal part of the step
   */
  void ComputeSMARTUpsilonAndNu(const PrimVec& p, const DualVec& Ap,
                                const double& A_norm, double& upsilon,
                                double& nu) const;

  /*!
   * \brief estimated bound for tangential part of step
   * \param[in] i - current iteration/size of subspace
   * \param[in] Z - subspace basis vectors
   * \param[in] A - reduced KKT-matrix
   * \param[in] gu - the reduced right-hand side for tangential part of step
   * \param[out] work - work vector to store tangential part of step
   * \param[out] upsilon - estimated bound on the tangential part of the step
   */
  void ComputeSMARTUpsilon(int i, const std::vector<Vec>& Z, 
                           const ublas::matrix<double>& A,
                           const ublas::vector<double>& gu,
                           PrimVec& work,
                           double& upsilon) const;
  
  /*!
   * \brief checks the Model Reduction Condition (3.12) from the SMART tests
   * \param[in] grad_dot_search - (gradient)^T * (search dir)
   * \param[in] pHp - the H-inner product, p^T * H * p, where H is Hessian
   * \param[in] theta - 2*theta is an estimate of the smallest Rayleigh quotient
   * \param[in] upsilon - upper bound on the tangential part of the step
   * \param[in] sigma - merit function penalty parameter scaling
   * \param[in] pi - merit function penalty parameter
   * \param[in] dual_norm0 - initial norm of the constraints
   * \param[in] dual_norm - norm of the linearized constraints using p
   */
  bool CheckSMARTModelReduction(
      const double& grad_dot_search, const double& pHp,
      const double& theta, const double& upsilon, const double& sigma,
      const double& pi, const double& dual_norm0, const double& dual_norm) const;


  /*!
   * \brief Calculates pi to satisfy model reduction when Test 2 is satisfied
   * \param[in] grad_dot_search - (gradient)^T * (search dir)
   * \param[in] pHp - the H-inner product, p^T * H * p, where H is Hessian
   * \param[in] theta - 2*theta is an estimate of the smallest Rayleigh quotient
   * \param[in] upsilon - upper bound on the tangential part of the step
   * \param[in] tau - parameter that controls the growth of pi
   * \param[in] dual_norm0 - initial norm of the constraints
   * \param[in] dual_norm - norm of the linearized constraints using p
   * \returns the new penalty parameter pi
   */
  double UpdatePenaltyParameter(
      const double& grad_dot_search, const double& pHp, const double& theta,
      const double& upsilon, const double& tau, const double& dual_norm0,
      const double& dual_norm) const;
};
// ==============================================================================
/*!
 * \class FISQPSolver
 * \brief Flexible Iterative Sequential Quadratic Programming method
 */
template <class Vec, class PrimVec, class DualVec>
class FISQPSolver : public IterativeSolver<Vec,Vec> {
 public:

  /*!
   * \brief default constructor
   */
  FISQPSolver();
  
  /*!
   * \brief class destructor
   */
  ~FISQPSolver() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

  /*!
   * \brief indicates how many user vectors are needed
   * \param[out] num_required - stores the number of user vectors required
   *
   * num_required["num_vec"] stores number of Vec vectors required
   * num_required["primal"] stores number of PrimVec vectors required
   * num_required["dual"] stores number of DualVec vectors required
   */
  void MemoryRequired(ptree& num_required) const;
  
  /*!
   * \brief solves the trust-region subproblem based on the inputs
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs (-gradient)
   * \param[out] x - the solution (assumed that x = 0.0 initially)
   * \param[in] mat_vec - object that defines matrix-vector product for Vec
   * \param[in] precond - object that defines preconditioner for Vec
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void Solve(const ptree & ptin, const Vec & b, Vec & x,
             MatrixVectorProduct<Vec> & mat_vec, Preconditioner<Vec> & precond,
             ptree & ptout, ostream & fout = cout);

  /*!
   * \brief resolves the trust-region problem using the existing subspace
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs (-gradient)
   * \param[out] x - the solution (assumed that x = 0.0 initially)
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void ReSolve(const ptree & ptin, const Vec & b, Vec & x, ptree & ptout,
               ostream & fout);
  
 private:

  /*
   * \brief solves the primal-dual and composite-step subspace problems
   * \param[in] iter - present size of the subspace
   * \param[in] radius - trust-region radius
   * \param[in] H - upper Hessenberg matrix from Arnoldi
   * \param[in] g - primal-dual subspace RHS
   * \param[in] g_tang - tangential subspace RHS
   * \param[in] ZtZ_prim - the primal part of the inner products Z^{T}*Z
   * \param[in] VtZ - the inner products V^{T}*Z
   * \param[in] VtZ_prim - the primal part of the inner products V^{T}*Z
   * \param[in] VtZ_dual - the dual part of the inner products V^{T}*Z
   * \param[in] VtV_dual - the dual part of the inner products V^{T}*V
   * \param[out] y - subspace solution of the primal-dual problem
   * \param[out] y_comp -subspace solution of the composite step problem
   * \param[out] y_tang - tangential component of y_comp
   * \param[out] y_norm - normal component of y_comp
   * \returns (beta, gamma, gamma_comp, neg_curv, step_violate) (see below)
   *
   * The returned tuple holds the primal-dual residual (beta), the primal-dual
   * constraint residual (gamma), the composite step constraint residual
   * (gamma_comp), the null-vector quality (null_qual), a boolean for whether
   * negative curvature was detected (neg_curv), and a boolean for whether the
   * primal-dual step exceeded the trust radius (step_violate), the predicted
   * reduction (pred), and the composite-step predicted reduction (pred_comp)
   */
  boost::tuple<double, double, double, double, bool, bool, double, double> 
  SolveSubspaceProblems(
      const int& iter, const double& radius, const ublas::matrix<double>& H,
      const ublas::vector<double>& g, const ublas::vector<double>& g_tang,
      const ublas::matrix<double>& ZtZ_prim, const ublas::matrix<double>& VtZ,
      const ublas::matrix<double>& VtZ_prim,
      const ublas::matrix<double>& VtZ_dual,
      const ublas::matrix<double>& VtV_dual, ublas::vector<double>& y,
      ublas::vector<double>& y_comp, ublas::vector<double>& y_tang,
      ublas::vector<double>& y_norm);

  /*
   * \brief write header data for FISQP output
   * \param[in,out] os - ostream class object for output
   * \param[in] tol - target tolerance for primal-dual problem
   * \param[in] res0 - the initial residual norm
   * \param[in] feas0 - the initial constraint norm
   */
  void WriteHeader(ostream& os, const double& tol, const double& res0,
                   const double& feas0);
  
  /*
   * \brief write data from a single FISQP iteration to output
   * \param[in,out] os - ostream class object for output
   * \param[in] iter - current iteration
   * \param[in] res0 - the initial residual norm
   * \param[in] feas0 - the initial constraint norm
   * \param[in] res - relative residual norm of primal-dual solution
   * \param[in] feas - relative constraint norm of primal-dual solution
   * \param[in] feas_comp - relative constraint norm of the composite-step sol.
   * \param[in] null_qual - measure of the quality of the null vector
   */
  void WriteHistory(ostream& os, const int& iter, const double& res,
                    const double& feas, const double& feas_comp,
                    const double& null_qual);
    
  int maxiter_; ///< maximum number of Krylov iterations
  int iters_; ///< size of subspace used (needed for second-order correction)
  std::vector<Vec> V_; ///< subspace basis for range of HZ
  std::vector<Vec> Z_; ///< primal solution basis
  boost::scoped_ptr<Vec> res_; ///< residual, b - Ax
  ublas::vector<double> y_; /// primal solution in the reduced space
  ublas::vector<double> y_old_; ///< previous iteration reduced-space solution
  ublas::vector<double> sn_; ///< sines for Givens rotations
  ublas::vector<double> cs_; ///< cosines for Givens rotations
  ublas::vector<double> g_; /// initial residuals in the reduced space
  ublas::vector<double> g_tang_; /// null-space projection problem rhs
  
  ublas::matrix<double> H_; /// upper Hessenberg matrix
  ublas::matrix<double> VtZ_; /// V^T * Z = VZprods_
  ublas::matrix<double> ZtZ_prim_; // primal part of inner products Z^{T}*Z
  ublas::matrix<double> VtV_dual_; // dual part of inner products V^{T}*V
  ublas::matrix<double> VtZ_prim_; // primal part of inner products V^{T}*Z
  ublas::matrix<double> VtZ_dual_; // dual part of inner products V^{T}*Z
};
// ==============================================================================
/*!
 * \class FLECSSolver
 * \brief Flexible Linearized-Equality-Constraint Subproblem Solver
 */
template <class Vec, class PrimVec, class DualVec>
class FLECSSolver : public IterativeSolver<Vec,Vec> {
 public:

  /*!
   * \brief default constructor
   */
  FLECSSolver();
  
  /*!
   * \brief class destructor
   */
  ~FLECSSolver() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

  /*!
   * \brief indicates how many user vectors are needed
   * \param[out] num_required - stores the number of user vectors required
   *
   * num_required["num_vec"] stores number of Vec vectors required
   * num_required["primal"] stores number of PrimVec vectors required
   * num_required["dual"] stores number of DualVec vectors required
   */
  void MemoryRequired(ptree& num_required) const;
  
  /*!
   * \brief solves the trust-region subproblem based on the inputs
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs (-gradient, -constraint)
   * \param[out] x - the solution (assumed that x = 0.0 initially)
   * \param[in] mat_vec - object that defines matrix-vector product for Vec
   * \param[in] precond - object that defines preconditioner for Vec
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void Solve(const ptree & ptin, const Vec & b, Vec & x,
             MatrixVectorProduct<Vec> & mat_vec, Preconditioner<Vec> & precond,
             ptree & ptout, ostream & fout = cout);

  /*!
   * \brief re-solves the trust-region problem using the existing subspace
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs (-gradient, -constraint)
   * \param[out] x - the solution (assumed that x = 0.0 initially)
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void ReSolve(const ptree & ptin, const Vec & b, Vec & x, ptree & ptout,
               ostream & fout);

  /*!
   * \brief computes a second-order correction for the primal step
   * \param[in] ptin - input parameters for solver
   * \param[in] ceq - the (negative) constraint value
   * \param[in] b - the rhs (-gradient, -constraint)
   * \param[out] x - the solution (only x.primal() is updated)
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history 
   */
  void Correct2ndOrder(const ptree & ptin, const DualVec & ceq, const Vec & b,
                       Vec & x, ptree & ptout, ostream & fout);
  
 private:

  /*
   * \brief solves the primal-dual and augmented-Lagrangian subspace problems
   * \param[in] iter - present size of the subspace
   * \param[in] radius - trust-region radius
   * \param[in] H - upper Hessenberg matrix from Arnoldi
   * \param[in] g - primal-dual subspace RHS
   * \param[in] mu - augmented-Lagrangian penalty parameter
   * \param[in] ZtZ_prim - the primal part of the inner products Z^{T}*Z
   * \param[in] VtZ - the inner products V^{T}*Z
   * \param[in] VtZ_prim - the primal part of the inner products V^{T}*Z
   * \param[in] VtZ_dual - the dual part of the inner products V^{T}*Z
   * \param[in] VtV_dual - the dual part of the inner products V^{T}*V
   * \param[out] y - subspace solution of the primal-dual problem
   * \param[out] y_aug -primal subspace solution of the composite step problem
   * \param[out] y_mult - dual subspace solution of the composite step problem
   * \returns (beta, gamma, beta_aug, gamma_aug, neg_curv, trust_active, pred, pred_aug) (see below)
   *
   * The returned tuple holds the primal-dual residual (beta), the primal-dual
   * constraint residual (gamma), the primal-dual primal residual (omega), the
   * aug-Lag step residual (beta_aug), the aug-Lag step constraint residual
   * (gamma_aug), a boolean for whether negative curvature is likely (neg_curv),
   * a boolean for whether the trust radius constraint is active (trust_active),
   * the predicted reduction (pred), and augmented-Lagrangian predicted
   * reduction (pred_aug)
   */
  boost::tuple<double, double, double, double, double, bool, bool, double,
               double> 
  SolveSubspaceProblems(
      const int& iter, const double& radius, const ublas::matrix<double>& H,
      const ublas::vector<double>& g, const double& mu,
      const ublas::matrix<double>& ZtZ_prim, const ublas::matrix<double>& VtZ,
      const ublas::matrix<double>& VtZ_prim,
      const ublas::matrix<double>& VtZ_dual,
      const ublas::matrix<double>& VtV_dual, ublas::vector<double>& y,
      ublas::vector<double>& y_aug, ublas::vector<double>& y_mult);

  /*
   * \brief write header data for FLECS output
   * \param[in,out] os - ostream class object for output
   * \param[in] tol - target tolerance for primal-dual problem
   * \param[in] res0 - the initial residual norm
   * \param[in] grad0 - the initial gradient norm
   * \param[in] feas0 - the initial constraint norm
   */
  void WriteHeader(ostream& os, const double& tol, const double& res0,
                   const double& grad0, const double& feas0);
  
  /*
   * \brief write data from a single FLECS iteration to output
   * \param[in,out] os - ostream class object for output
   * \param[in] iter - current iteration
   * \param[in] res - relative residual norm of primal-dual solution
   * \param[in] grad - relative gradient norm of primal-dual solution
   * \param[in] feas - relative constraint norm of primal-dual solution
   * \param[in] feas_aug - relative constraint norm of the aug-Lag step
   * \param[in] pred - predicted reduction in objective using FGMRES step
   * \param[in] pred_aug - predicted reduction in objective using aug-Lag step
   */
  void WriteHistory(ostream& os, const int& iter, const double& res,
                    const double& grad, const double& feas,
                    const double& feas_aug, const double& pred,
                    const double& pred_aug);
    
  int maxiter_; ///< maximum number of Krylov iterations
  int iters_; ///< size of subspace used (needed for second-order correction)
  double mu_; ///< penalty parameter (needed for resolve)
  std::vector<Vec> V_; ///< subspace basis for range of HZ
  std::vector<Vec> Z_; ///< primal solution basis
  boost::scoped_ptr<Vec> res_; ///< residual, b - Ax
  ublas::vector<double> y_; /// primal solution in the reduced space
  ublas::vector<double> g_; /// initial residuals in the reduced space
  
  ublas::matrix<double> H_; /// upper Hessenberg matrix
  ublas::matrix<double> VtZ_; /// V^T * Z = VZprods_
  ublas::matrix<double> ZtZ_prim_; // primal part of inner products Z^{T}*Z
  ublas::matrix<double> VtV_dual_; // dual part of inner products V^{T}*V
  ublas::matrix<double> VtZ_prim_; // primal part of inner products V^{T}*Z
  ublas::matrix<double> VtZ_dual_; // dual part of inner products V^{T}*Z
};
// ==============================================================================
/*!
 * \class BPDSolver
 * \brief Balanced Primal-Dual Solver
 */
template <class Vec, class PrimVec, class DualVec>
class BPDSolver : public IterativeSolver<Vec,Vec> {
 public:

  /*!
   * \brief default constructor
   */
  BPDSolver();
  
  /*!
   * \brief class destructor
   */
  ~BPDSolver() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

  /*!
   * \brief indicates how many user vectors are needed
   * \param[out] num_required - stores the number of user vectors required
   *
   * num_required["num_vec"] stores number of Vec vectors required
   * num_required["primal"] stores number of PrimVec vectors required
   * num_required["dual"] stores number of DualVec vectors required
   */
  void MemoryRequired(ptree& num_required) const;
  
  /*!
   * \brief solves a primal-dual system in a balanced way
   * \param[in] ptin - input parameters for solver
   * \param[in] b - the rhs (-gradient and -constraints)
   * \param[out] x - the solution
   * \param[in] mat_vec - object that defines primal-dual-matrix-vector product
   * \param[in] precond - object that defines preconditioner
   * \param[out] ptout - output information from solver
   * \param[in] fout - object for writing the convergence history
   *
   * Given scalings provided in ptin, this solver insures that both the primal
   * and the dual problems are inexactly solved (both have their residuals
   * reduced equally).
   */
  void Solve(const ptree & ptin, const Vec & b, Vec & x,
             MatrixVectorProduct<Vec> & mat_vec, Preconditioner<Vec> & precond,
             ptree & ptout, ostream & fout = cout);
  
 private:

  /*
   * \brief write header data for BPD output
   * \param[in,out] os - ostream class object for output
   * \param[in] tol - target tolerance for primal-dual problem
   * \param[in] res0 - the initial residual norm of problem
   * \param[in] grad0 - the initial gradient norm
   * \param[in] feas0 - the initial constraint norm
   */
  void WriteHeader(ostream& os, const double& tol, const double& res0,
                   const double& grad0, const double& feas0);
  
  /*
   * \brief write data from a single BPD iteration to output
   * \param[in,out] os - ostream class object for output
   * \param[in] iter - current iteration
   * \param[in] res - relative residual norm of primal-dual solution
   * \param[in] grad - relative gradient norm
   * \param[in] feas - relative constraint norm
   */
  void WriteHistory(ostream& os, const int& iter, const double& res,
                    const double& grad, const double& feas);
    
  int maxiter_; ///< maximum number of Krylov iterations
  int iters_; ///< size of subspace used (needed for second-order correction)
  std::vector<Vec> V_; ///< subspace basis for range of HZ
  std::vector<Vec> Z_; ///< primal solution basis
  ublas::vector<double> y_; /// primal solution in the reduced space
  ublas::vector<double> g_; /// initial residuals in the reduced space
  ublas::matrix<double> H_; /// upper Hessenberg matrix
  ublas::matrix<double> VtV_prim_; // primal part of inner products V^{T}*V
  ublas::matrix<double> VtV_dual_; // dual part of inner products V^{T}*V
};
// ==============================================================================

/*!
 * \brief sign transfer function
 * \param[in] x - value having sign prescribed
 * \param[in] y - value that defined the sign
 */
double sign(const double & x, const double & y);

/*!
 * \brief determines the perturbation parameter for Jacobian-free product
 * \param[in] eval_at_norm - L2 norm of vector where Jacobian evaluated
 * \param[in] mult_by_norm - L2 norm of vector that is being multiplied
 * \returns the perturbation parameter for Jacobian-free product
 */
double CalcEpsilon(const double & eval_at_norm,
                   const double & mult_by_norm);

/*!
 * \brief returns the eigenvalues of the symmetric part of a square matrix
 * \param[in] n - size of the square matrix (actual size of A may be bigger)
 * \param[in] A - matrix stored in dense format; not necessarily symmetric
 * \param[out] eig - the eigenvalues in ascending order
 *
 * The matrix A is stored in dense format and is not assumed to be exactly
 * symmetric.  The eigenvalues are found by calling a LAPACK routine, which is
 * given 0.5*(A^T + A) as the input matrix, not A itself.
 */
void eigenvalues(const int & n, const ublas::matrix<double> & A,
                 ublas::vector<double> & eig);

/*!
 * \brief returns the eigenvalues and eigenvectors of a symmetric matrix
 * \param[in] n - size of the square matrix (actual size of A may be bigger)
 * \param[in] A - matrix stored in dense format; not necessarily symmetric
 * \param[out] eig - the eigenvalues in ascending order
 * \param[out] E - the eigenvectors corresponding to eig
 *
 * The matrix A is stored in dense format and is not assumed to be exactly
 * symmetric.  The eigenvalues are found by calling a LAPACK routine, which is
 * given 0.5*(A^T + A) as the input matrix, not A itself.
 */
void eigenvaluesAndVectors(const int & n, const ublas::matrix<double> & A,
                           ublas::vector<double> & eig,
                           ublas::matrix<double> & E);

/*!
 * \brief returns the Q-R factorization of the given matrix A
 * \param[in] nrow - number of rows in matrix to consider: nrow <= A.size1()
 * \param[in] ncol - number of columns in matrix to consider: ncol <= A.size2()
 * \param[in] A - rectangular matrix stored in dense format
 * \param[out] QR - QR decomposition (see storage details below)
 *
 * This routine is a front end for the LAPACK routine DGEQRF.  As such, it
 * follows the same storage scheme for QR.  QR itself is a 1-d array.  The first
 * nrow*ncol part of QR stores Q and R in column-major (i.e. FORTRAN) ordering.
 * R is stored in the upper triangular part of QR.  Q is stored as elementary
 * reflectors in the lower half of A and the factors for these reflectors are
 * stored at the end of QR in QR(nrow*ncol:(nrow+1)*ncol-1).  Users should not
 * need to know these details, since operations with Q and R are provided by
 * other routines.
 */
void factorQR(const int & nrow, const int & ncol,
              const ublas::matrix<double> & A, ublas::vector<double> & QR);

/*!
 * \brief solves the upper triangluar problem Rx=b, where R is from a QR-fac.
 * \param[in] nrow - number of rows in the QR product
 * \param[in] ncol - number of columns in the QR product (R is ncol X ncol)
 * \param[in] QR - the QR factorization of some matrix
 * \param[in] b - the rhs vector in the system
 * \param[out] x - the solution vector
 * \param[in] transpose - if true, solve R^{T}x = b
 */
void solveR(const int & nrow, const int & ncol, ublas::vector<double> & QR,
            const ublas::vector<double> & b, ublas::vector<double> & x,
            const bool & transpose = false);

/*!
 * \brief applies the orthogonal matrix Q, from a QR factorization, to b
 * \param[in] nrow - number of rows in the vector b
 * \param[in] ncol - number of orthogonal vectors in Q
 * \param[in] QR - the matrix Q stored as part of a QR factorization
 * \param[in] b - the vector that Q is multiplying from the left
 * \param[out] x - the result of the product: x = Q*b, or x = Q^T*b
 * \param[in] transpose - if true, returns Q^T b
 */
void applyQ(const int & nrow, const int & ncol, ublas::vector<double> & QR,
            const ublas::vector<double> & b, ublas::vector<double> & x,
            const bool & transpose = false);

/*!
 * \brief returns the Cholesky factorization of a SPD matrix A
 * \param[in] n - number of rows/columns in matrix
 * \param[in] A - matrix stored in dense format
 * \param[out] UTU - Cholesky factorization (see storage details below)
 *
 * This routine is a front end for the LAPACK routine DPOTRF.  As such, it
 * follows the same storage scheme for UTU.  Users should not need to know the
 * details, since operations with U and U^T are provided by other routines.
 */
void factorCholesky(const int & n, const ublas::matrix<double> & A,
                    ublas::vector<double> & UTU);

/*!
 * \brief solve triangular system with U or U^T from factorCholesky
 * \param[in] n - number of rows/columns in matrix
 * \param[out] UTU - Cholesky factorization
 * \param[in] b - the rhs vector
 * \param[out] x - the solution of U^T x = b, or U x = b
 * \param[in] transpose - if true, solves U^T x = b
 */
void solveU(const int & n, ublas::vector<double> & UTU,
            const ublas::vector<double> & b, ublas::vector<double> & x,
            const bool & transpose = false);

/*!
 * \brief computes the singular value decomposition of A
 * \param[in] nrow - number of rows in matrix to consider: nrow <= A.size1()
 * \param[in] ncol - number of columns in matrix to consider: ncol <= A.size2()
 * \param[in] A - rectangular matrix stored in dense format
 * \param[out] Sigma - the singular values of A, stored in descending order
 * \param[out] U - the first ncol left singular vectors of A + ortho vectors
 * \param[out] VT - the ncol right singular vectors of A, transposed
 * \param[in] All_of_U - if true, U is nrow x nrow orthonormal matrix
 */
void computeSVD(const int & nrow, const int & ncol,
                const ublas::matrix<double> & A,
                ublas::vector<double> & Sigma, ublas::vector<double> & U, 
                ublas::vector<double> & VT, const bool& All_of_U = false);

/*!
 * \brief applies a Givens rotation to a 2-vector
 * \param[in] s - sine of the Givens rotation angle
 * \param[in] c - cosine of the Givens rotation angle
 * \param[in,out] h1 - first element of 2x1 vector being transformed
 * \param[in,out] h2 - second element of 2x1 vector being transformed
 */
void applyGivens(const double & s, const double & c,
		 double & h1, double & h2);

/*!
 * \brief generates the Givens rotation matrix for a given 2-vector
 * \param[in,out] dx - element of 2x1 vector being transformed
 * \param[in,out] dy - element of 2x1 vector being set to zero
 * \param[in,out] s - sine of the Givens rotation angle
 * \param[in,out] c - cosine of the Givens rotation angle
 *
 * Based on givens() of SPARSKIT, which is based on p.202 of
 * "Matrix Computations" by Golub and van Loan.
 */
void generateGivens(double & dx, double & dy, double & s, double & c);

/*!
 * \brief finds the solution of a general linear system (single rhs)
 * \param[in] n - size of the reduced system
 * \param[in] A - general (non-symmetric) system matrix
 * \param[in] rhs - right-hand side of the reduced system
 * \param[out] x - solution of the reduced system
 */
void solveReduced(const int & n, const ublas::matrix<double> & A,
                  const ublas::vector<double> & rhs,
                  ublas::vector<double> & x);

/*!
 * \brief finds the solution of a general linear system with multiple rhs
 * \param[in] n - size of the reduced system
 * \param[in] A - general (non-symmetric) system matrix
 * \param[in] nrhs - number of right-hand-sides
 * \param[in] RHS - right-hand side(s) of the reduced system
 * \param[out] X - solution(s) of the reduced system
 */
void solveReducedMultipleRHS(const int & n, const ublas::matrix<double> & A,
                             const int & nrhs, const ublas::matrix<double> & RHS,
                             ublas::matrix<double> & X); 

/*!
 * \brief finds the solution of the upper triangular system Hsbg*x = rhs
 * \param[in] n - size of the reduced system
 * \param[in] Hsbg - upper triangular matrix
 * \param[in] rhs - right-hand side of the reduced system
 * \param[out] x - solution of the reduced system
 *
 * \pre the upper Hessenberg matrix has been transformed into a
 * triangular matrix.
 */
void solveReducedHessenberg(const int & n, const ublas::matrix<double> & Hsbg,
                            const ublas::vector<double> & rhs,
                            ublas::vector<double> & x);

/*!
 * \brief solves the reduced-space trust-region problem
 * \param[in] n - size of the reduced space
 * \param[in] H - reduced-space Hessian
 * \param[in] radius - trust region radius
 * \param[in] g - gradient in the reduced space
 * \param[out] y - solution to reduced-space trust-region problem
 * \param[in,out] lambda - entry: a guess for Lagrange mult. exit: the mult value
 * \param[out] - predicted decrease in the objective
 *
 * This assumes the reduced space objective is in the form g'*x + 0.5*x'*H*x.
 * Furthermore, the case g = 0 is not handled presently.
 */
void solveTrustReduced(const int & n, const ublas::matrix<double> & H,
                       const double & radius, const ublas::vector<double> & g,
                       ublas::vector<double> & y, double & lambda,
                       double & pred);
/*!
 * \brief function for solveTrustReduced
 * \param[in] n - size of the reduced space
 * \param[in] H - reduced-space Hessian
 * \param[in] g - gradient in the reduced space
 * \param[in] lambda - value of lambda at which to evaluate function
 * \param[in] radius - trust region radius
 * \returns a boost tuple of the step, function, and function's derivative
 */
boost::tuple<ublas::vector<double>, double, double> 
trustFunction(const int& n, const ublas::matrix<double> & H,
              const ublas::vector<double> & g, const double& lambda,
              const double& radius);

/*!
 * \brief solves an underdetermined system using the minimum-norm solution
 * \param[in] nrow - the number of rows in A
 * \param[in] ncol - the number of columns in A
 * \param[in] A - the matrix of the system
 * \param[in] b - the right-hand side of the under-determined system
 * \param[out] x - the solution
 */
void solveUnderdeterminedMinNorm(const int & nrow, const int & ncol,
                                 const ublas::matrix<double> & A,
                                 const ublas::vector<double> & b,
                                 ublas::vector<double> & x);

/*!
 * \brief solves an underdetermined system using the minimum-norm solution
 * \param[in] nrow - the number of rows in U*Simga*V^T
 * \param[in] ncol - the number of columns in U*Simga*V^T
 * \param[in] Sigma - the singular values stored in a vector in descending order
 * \param[in] U - the left left singular vectors
 * \param[in] VT - the first nrow right singular vectors, transposed
 * \param[in] b - the right-hand side of the under-determined system
 * \param[out] x - the solution
 */
void solveUnderdeterminedMinNorm(const int & nrow, const int & ncol,
                                 const ublas::vector<double> & Sigma,
                                 const ublas::matrix<double> & U,
                                 const ublas::matrix<double> & VT,
                                 const ublas::vector<double> & b,
                                 ublas::vector<double> & x);

/*!
 * \brief solves a least-squares problem
 * \param[in] nrow - the number of equations
 * \param[in] ncol - the number of unknowns
 * \param[in] A - matrix storing equation coefficients
 * \param[in] b - the right-hand side of the least-squares problem
 * \param[out] x - the solution
 */
void solveLeastSquares(const int & nrow, const int & ncol,
                       const ublas::matrix<double>& A,
                       const ublas::vector<double>& b,
                       ublas::vector<double>& x);

/*!
 * \brief solves a Least-squares problem with solution constrained to a sphere
 * \param[in] nrow - the number of rows in U*Simga*V^T
 * \param[in] ncol - the number of columns in U*Simga*V^T
 * \param[in] radius - the radius of the sphere/trust region
 * \param[in] Sigma - the singular values stored in a vector in descending order
 * \param[in] U - the first ncol left singular vectors
 * \param[in] VT - the right singular vectors, transposed
 * \param[in] b - the right-hand side of the least-squares problem
 * \param[out] x - the solution
 * \returns the Lagrange multiplier
 */
double solveLeastSquaresOverSphere(const int & nrow, const int & ncol,
                                   const double & radius,
                                   const ublas::vector<double> & Sigma,
                                   const ublas::matrix<double> & U, 
                                   const ublas::matrix<double> & VT,
                                   const ublas::vector<double> & b,
                                   ublas::vector<double> & x);

/*!
 * \brief computes the residual norm for the FITR method
 * \param[in] n - size of the reduced space
 * \param[in] H - upper Hessenberg matrix from A*Z = V*Hsbg
 * \param[in] B - (n+1)*n matrix from V^{T} * Z
 * \param[in] g - gradient in the reduced space
 * \param[in] y - solution to reduced-space trust-region problem
 * \param[in] lambda - the Lagrange multiplier value
 * \returns the residual norm of the solution
 */
double trustResidual(const int & n, const ublas::matrix<double> & H,
                     const ublas::matrix<double> & B,
                     const ublas::vector<double> & g,
                     const ublas::vector<double> & y,
                     const double & lambda);

/*!
 * \brief checks for nonzeros on the diagonal of a matrix
 * \param[in] n - size of the reduced system
 * \param[in] T - triangular matrix
 *
 * \pre to be useful, T must be a triangular matrix
 */
bool triMatixInvertible(const int & n, const ublas::matrix<double> & T);

/*!
 * \brief writes header information for a Krylov residual history
 * \param[in,out] os - ostream class object for output
 * \param[in] solver - string describing the solver
 * \param[in] restol - the target tolerance to solve to
 * \param[in] resinit - the initial residual norm (absolute)
 *
 * \pre the ostream object os should be open
 */
void writeKrylovHeader(ostream & os, const std::string & solver,
		       const double & restol, const double & resinit);

/*!
 * \brief writes header information for a Krylov residual history
 * \param[in,out] os - ostream class object for output
 * \param[in] solver - string describing the solver
 * \param[in] restol - the target tolerance to solve to
 * \param[in] resinit - the initial residual norm (absolute)
 * \param[in] col_header - a formatted string listing the column headers
 *
 * \pre the ostream object os should be open
 */
void writeKrylovHeader(ostream & os, const std::string & solver,
		       const double & restol, const double & resinit,
                       const boost::format & col_heder);

/*!
 * \brief writes residual convergence data for one iteration to a stream
 * \param[in,out] os - ostream class object for output
 * \param[in] iter - current iteration
 * \param[in] res - the (absolute) residual norm value
 * \param[in] resinit - the initial residual norm
 *
 * \pre the ostream object os should be open
 */
void writeKrylovHistory(ostream & os, const int & iter,
			const double & res, const double & resinit);

// ==============================================================================

/*!
 * \brief Modified Gram-Schmidt orthogonalization
 * \author Based on Kesheng John Wu's mgsro subroutine in Saad's SPARSKIT
 * \tparam Vec - a generic vector class
 * \param[in] i - index indicating which vector is orthogonalized
 * \param[in,out] Hsbg - the upper Hessenberg matrix begin updated
 * \param[in,out] w - the (i+1)th vector of w is orthogonalized against
 *                    the previous vectors w[0:i]
 *
 * \pre the vectors w[0:i] are orthonormal
 * \post the vectors w[0:i+1] are orthonormal
 *
 * Reothogonalization is performed if the cosine of the angle between
 * w[i+1] and w[k], k < i+1, is greater than 0.98.  The norm of the "new"
 * vector is kept in nrm0 and updated after operating with each vector
 *
 * If i < 0, the vector w[i+1] is simply normalized.
 */
template <class Vec>
void modGramSchmidt(int i, ublas::matrix<double> & Hsbg, vector<Vec> & w);

/*!
 * \brief Modified Gram-Schmidt orthogonalization (no Hessenberg update)
 * \author Based on Kesheng John Wu's mgsro subroutine in Saad's SPARSKIT
 * \tparam Vec - a generic vector class
 * \param[in] i - index indicating which vector is orthogonalized
 * \param[in,out] w - the (i+1)th vector of w is orthogonalized against
 *                    the previous vectors w[0:i]
 *
 * \pre the vectors w[0:i] are orthonormal
 * \post the vectors w[0:i+1] are orthonormal
 *
 * Reothogonalization is performed if the cosine of the angle between
 * w[i+1] and w[k], k < i+1, is greater than 0.98.  The norm of the "new"
 * vector is kept in nrm0 and updated after operating with each vector
 *
 * If i < 0, the vector w[i+1] is simply normalized.
 */
template <class Vec>
void modGramSchmidt(int i, vector<Vec> & w);

/*!
 * \brief Estimate the largest magnitude eigenvalue of a symmetric matrix
 * \tparam Vec - a generic vector class
 * \param[in] m - maximum number of iterations
 * \param[in] tol - tolerance between successive esitmates
 * \param[in] mat_vec - object that defines matrix-vector product for Vec
 * \param[out] iters - total number of iterations
 * \param[in] fout - object for writing the convergence history
 * \returns the estimate of max eig(A)
 *
 * Uses the symmetric Lanczos algorithm to find an estimate of the largest
 * eigenvalue of A (defined implicitly through mat_vec).
 */
template <class Vec>
double Lanczos(int m, double tol, MatrixVectorProduct<Vec> & mat_vec,
               int & iters, ostream & fout = cout);

/*!
 * \brief Right-preconditioned Generalized Minimal RESidual method
 * \tparam Vec - a generic vector class
 * \param[in] m - maximum size of the search subspace
 * \param[in] tol - tolerance with which to solve the system
 * \param[in] b - the right hand size vector
 * \param[in,out] x - on entry the intial guess, on exit the solution
 * \param[in] mat_vec - object that defines matrix-vector product for Vec
 * \param[out] iters - total number of iterations or preconditioner calls
 * \param[in] precond - object that defines (right) preconditioner for Vec
 * \param[in] fout - object for writing the convergence history 
 * \param[in] check_res - determines if the residual should be checked at the end
 * \param[in] dynamic - if true, the matvec accuracy is varied dynamically
 */
template <class Vec>
void GMRES(int m, double tol, const Vec & b, Vec & x,
           MatrixVectorProduct<Vec> & mat_vec,
           Preconditioner<Vec> & precond, int & iters, 
           ostream & fout = cout, const bool & check_res = true,
           const bool & dynamic = false);

/*!
 * \brief Flexible Generalized Minimal RESidual method
 * \tparam Vec - a generic vector class
 * \param[in] m - maximum size of the search subspace
 * \param[in] tol - tolerance with which to solve the system
 * \param[in] b - the right hand size vector
 * \param[in,out] x - on entry the intial guess, on exit the solution
 * \param[in] mat_vec - object that defines matrix-vector product for Vec
 * \param[out] iters - total number of iterations or preconditioner calls
 * \param[in] precond - object that defines preconditioner for Vec
 * \param[in] fout - object for writing the convergence history 
 * \param[in] check_res - determines if the residual should be checked at the end
 * \param[in] dynamic - if true, the matvec accuracy is varied dynamically
 * \param[out] res - used to return the (absolute!) residual norm, if needed
 * \param[out] sig - the smallest singular value of the upper Hessenberg matrix
 */
template <class Vec>
void FGMRES(int m, double tol, const Vec & b, Vec & x,
	    MatrixVectorProduct<Vec> & mat_vec,
	    Preconditioner<Vec> & precond, int & iters, 
            ostream & fout = cout, const bool & check_res = true,
            const bool & dynamic = false, const boost::scoped_ptr<double> & res
            = boost::scoped_ptr<double>(), const boost::scoped_ptr<double> & sig
            = boost::scoped_ptr<double>());

/*!
 * \brief Flexible Full Orthogonalization Method (FFOM)
 * \tparam Vec - a generic vector class
 * \param[in] m - maximum size of the search subspace
 * \param[in] tol - tolerance with which to solve the system
 * \param[in] b - the right hand size vector
 * \param[in,out] x - on entry the intial guess, on exit the solution
 * \param[in] mat_vec - object that defines matrix-vector product for Vec
 * \param[out] iters - total number of iterations or preconditioner calls
 * \param[in] precond - object that defines preconditioner for Vec
 * \param[in] fout - object for writing the convergence history 
 * \param[in] check_res - determines if the residual should be checked at the end
 * \param[in] dynamic - if true, the matvec accuracy is varied dynamically
 */
template <class Vec>
void FFOM(int m, double tol, const Vec & b, Vec & x,
          MatrixVectorProduct<Vec> & mat_vec,
          Preconditioner<Vec> & precond, int & iters, 
          ostream & fout = cout, const bool & check_res = true,
          const bool & dynamic = false);

/*!
 * \brief Steihaug trust-region CG solver
 * \tparam Vec - a generic vector class
 * \param[in] m - maximum size of the search subspace
 * \param[in] tol - tolerance with which to solve the system
 * \param[in] radius - trust-region radius
 * \param[in] b - the right hand size vector
 * \param[out] x - solution (assume x = 0 for initial guess)
 * \param[in] mat_vec - object that defines matrix-vector product for Vec
 * \param[in] precond - object that defines preconditioner for Vec
 * \param[out] iters - total number of preconditioner calls
 * \param[out] pred - predicted decrease in the objective
 * \param[out] active - true if the trust-region constraint is active
 * \param[in] fout - object for writing the convergence history 
 * \param[in] check_res - determines if the residual should be checked at the end
 * \param[in] dynamic - if true, the matvec accuracy is varied dynamically
 */
template <class Vec>
void SteihaugCG(int m, double tol, double radius, const Vec & b, Vec & x,
                MatrixVectorProduct<Vec> & mat_vec,
                Preconditioner<Vec> & precond, int & iters,
                double & pred, bool & active,
                ostream & fout = cout, const bool & check_res = true,
                const bool & dynamic = false);

/*!
 * \brief Steihaug variant of Full Orthogonalization Method (FOM)
 * \tparam Vec - a generic vector class
 * \param[in] m - maximum size of the search subspace
 * \param[in] tol - tolerance with which to solve the system
 * \param[in] radius - trust-region radius
 * \param[in] b - the right hand size vector
 * \param[in,out] x - on entry the intial guess, on exit the solution
 * \param[in] mat_vec - object that defines matrix-vector product for Vec
 * \param[out] iters - total number of iterations or preconditioner calls
 * \param[out] pred - predicted decrease in the objective
 * \param[out] active - true if the trust-region constraint is active 
 * \param[in] precond - object that defines preconditioner for Vec
 * \param[in] fout - object for writing the convergence history 
 * \param[in] check_res - determines if the residual should be checked at the end
 * \param[in] dynamic - if true, the matvec accuracy is varied dynamically
 */
template <class Vec>
void SteihaugFOM(int m, double tol, double radius, const Vec & b, Vec & x,
                 MatrixVectorProduct<Vec> & mat_vec,
                 Preconditioner<Vec> & precond, int & iters, double & pred,
                 bool & active, ostream & fout = cout,
                 const bool & check_res = true, const bool & dynamic = false);

/*!
 * \brief Minimal RESidual method of 
 * \tparam Vec - a generic vector class
 * \param[in] m - maximum size of the search subspace
 * \param[in] tol - tolerance with which to solve the system
 * \param[in] b - the right hand size vector
 * \param[in,out] x - on entry the intial guess, on exit the solution
 * \param[in] mat_vec - object that defines matrix-vector product for Vec
 * \param[out] iters - total number of preconditioner calls (iterations + 1)
 * \param[in] precond - object that defines preconditioner for Vec
 * \param[in] fout - object for writing the convergence history 
 * \param[in] check_res - determines if the residual should be checked at the end
 * \param[in] dynamic - if true, the matvec accuracy is varied dynamically
 *
 * This is an implementation of MINRES (see C. C. Paige and M. A. Saunders
 * (1975), Solution of sparse indefinite systems of linear equations, SIAM
 * J. Numer. Anal. 12(4), pp. 617-629).  The code is based on the F90 version
 * developed by the Stanford Systems Optimization Laboratory.
 */
template <class Vec>
void MINRES(int m, double tol, const Vec & b, Vec & x,
	    MatrixVectorProduct<Vec> & mat_vec,
	    Preconditioner<Vec> & precond, int & iters, 
            ostream & fout = cout, const bool & check_res = true,
            const bool & dynamic = false);

/*!
 * \brief Flexible Iterative Trust Region method
 * \tparam Vec - a generic vector class
 * \param[in] m - maximum size of the search subspace
 * \param[in] tol - tolerance with which to solve the system
 * \param[in] radius - trust-region radius
 * \param[in] b - the right hand size vector
 * \param[out] x - the solution (assumed that x = 0.0 initially for this solver)
 * \param[in] mat_vec - object that defines matrix-vector product for Vec
 * \param[in] precond - object that defines preconditioner for Vec
 * \param[out] iters - total number of iterations or preconditioner calls
 * \param[out] pred - predicted decrease in the objective
 * \param[out] active - true if the trust-region constraint is active
 * \param[in] fout - object for writing the convergence history 
 * \param[in] check - if true, a couple of checks are performed at end
 * \param[in] dynamic - if true, the matvec accuracy is varied dynamically
 */
template <class Vec>
void FITR_old(int m, double tol, double radius, const Vec & b, Vec & x,
              MatrixVectorProduct<Vec> & mat_vec,
              Preconditioner<Vec> & precond, int & iters, double & pred,
              bool & active, ostream & fout = cout, const bool & check = true,
              const bool & dynamic = false);

// ==============================================================================
/*!
 * \class TRISQP
 * \brief Trust-Region Iterative Sequential-Quadratic Programming method
 *
 * This method is implemented as a class --- rather than a function like the
 * other Krylov solvers --- because of the need to reuse subspace information
 * during second-order corrections.
 */
template <class Vec, class PrimVec, class DualVec>
class TRISQP {
 public:

  /*!
   * \brief default constructor
   */
  TRISQP();
  
  /*!
   * \brief class destructor
   */
  ~TRISQP() {}

  /*!
   * \brief sets the maximum number of subspace iterations, and sizes vectors
   * \param[in] m - maximum number of Krylov subspace iterations
   */
  void SubspaceSize(int m);

#if 0
  /*!
   * \brief constructs a basis for the normal and tangential subproblems
   * \param[in] tol - tol for the min. norm problem and null space
   */
  void BuildBasis(double tol, const Vec & b, Vec & x,
                  MatrixVectorProduct<Vec> & mat_vec,
                  Preconditioner<Vec> & precond, int & iters, ostream & fout);
#endif
  
  /*!
   * \brief run Trust-Region Iterative Sequential-Quadratic Programming method
   * \param[in] grad_tol - desired tolerance for the gradient of the Lagrangian
   * \param[in] feas_tol - desired tolerance for the constraints to satisfy
   * \param[in] radius - trust-region radius
   * \param[in] b - negative gradient of the Lagrangian
   * \param[out] x - the solution (assumed that x = 0.0 initially)
   * \param[in] mat_vec - object that defines matrix-vector product for Vec
   * \param[in] precond - object that defines preconditioner for Vec
   * \param[out] iters - total number of iterations or preconditioner calls
   * \param[in/out] mu - the penalty parameter for the Augmented Lagragian
   * \param[out] pred_opt - predicted decrease in Lagrangian
   * \param[out] pred_feas - predicted increase in feasibility
   * \param[out] active - true if the trust-region constraint is active
   * \param[in] fout - object for writing the convergence history 
   * \param[in] check - if true, a couple of checks are performed at end
   * \param[in] dynamic - if true, the matvec accuracy is varied dynamically
   */
  void Solve(double grad_tol, double feas_tol, double radius,
             const Vec & b, Vec & x, MatrixVectorProduct<Vec> & mat_vec,
             Preconditioner<Vec> & precond, int & iters, double & mu,
             double & pred_opt, double & pred_feas, bool & active,
             ostream & fout = cout, const bool & check = true,
             const bool & dynamic = false);

  /*!
   * \brief computes a second-order correction to overcome the Marotos effect
   * \param[in] ceq - the constraint residual evaluated after initial step
   * \param[in/out] x - the primal step that is added to
   * \param[in] radius - trust-region radius
   * \param[in/out] mu - the penalty parameter for the Augmented Lagragian
   * \param[out] pred_opt - predicted decrease in Lagrangian
   * \param[out] pred_feas - predicted increase in feasibility
   */
  void Correct2ndOrder(const DualVec & ceq, PrimVec & x, double radius,
                       double & mu, double & pred_opt, double & pred_feas);
  
 protected:

  int maxiter_; /// maximum number of Krylov iterations
  int iters_; /// number of Krylov iterations used in Solve()
  vector<PrimVec> V_; /// primal residual subspace
  vector<PrimVec> VTilde_; /// primal solution subspace
  vector<PrimVec> Z_; /// basis for constraint Jacobian range
  vector<DualVec> Lambda_; /// dual residual subspace
  Vec r_; /// residual vector
  
  ublas::vector<double> f_, g_; /// initial residuals in the reduced space
  ublas::vector<double> y_; /// primal solution in the reduced space
  ublas::vector<double> sig_; /// dual solution in the reduced space
  ublas::matrix<double> B_; /// H * VTilde_ + A^T * LambdaTilde_ = V_ * B_
  ublas::matrix<double> C_; /// A * VTilde = Lambda * C_
  ublas::matrix<double> D_; /// A^T * LambdaTilde_ = Z_ * D_
  ublas::matrix<double> Vprods_; /// VTilde^T * V = Vprods_
  ublas::matrix<double> VZprods_; /// VTilde^T * Z = VZprods_
};
  
} // namespace kona

// the implementations
#include "./krylov_def.hpp"
