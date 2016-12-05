/**
 * \file aerostruct.hpp
 * \brief header file for AeroStruct
 * \author Jason Hicken <jason.hicken@gmail.com>, Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

#pragma once

#include <math.h>

#include <ostream>
#include <iostream>
#include <fstream>
#include <string>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "./Quasi1DEuler/nozzle.hpp"
#include "./Quasi1DEuler/inner_prod_vector.hpp"
#include "./Quasi1DEuler/quasi_1d_euler.hpp"
#include "./Quasi1DEuler/exact_solution.hpp"
#include "./LECSM/lecsm.hpp"

#include "./constants.hpp"
#include "./krylov.hpp"

// forward declarations
class AeroStructProduct;
class AeroStructTransposeProduct;
class AeroStructPrecond;
class AeroStructTransposePrecond;

// ======================================================================

/*!
 * \class AeroStructMDA
 * \brief defines a coupled aero-structural system
 */
class AeroStructMDA {
public:

  /*!
   * \brief empty constructor
   */
  AeroStructMDA() {}

  /*!
   * \brief default constructor
   * \param[in] euler_solver - a Quasi1DEuler solver (defines product)
   */
  AeroStructMDA(int num_nodes, int order):
      u_(6*num_nodes,0.0),
      v_(6*num_nodes,0.0),
      cfd_(num_nodes, order),
      csm_(num_nodes) {
    num_nodes_ = num_nodes;
    order_ = order;
    scale_cfd_ = 1.0;
    scale_csm_ = 1.0;
  }

  /*!
   * \brief alternative class constructor for geometries defined by a Bspline
   * \param[in] num_nodes - number of nodes
   * \param[in] order - order of accuracy of CFD solver
   * \param[in] nozzle - BsplineNozzle the defines the area
   */
  AeroStructMDA(int num_nodes, int order, Nozzle & nozzle):
      u_(6*num_nodes,0.0),
      v_(6*num_nodes,0.0),
      cfd_(num_nodes, order),
      csm_(num_nodes) {
    num_nodes_ = num_nodes;
    order_ = order;
    scale_cfd_ = 1.0;
    scale_csm_ = 1.0;
    nozzle_ = &nozzle;
  }

  ~AeroStructMDA() {} ///< class destructor

  /*!
   * \brief extract the MDA solution vector
   */
  InnerProdVector & get_u() { return u_; }

  /*!
   * \brief extract the MDA residual vector
   */
  InnerProdVector & get_res() { return v_; }

  /*!
   * \brief define the MDA solution vector
   */
  void set_u(const InnerProdVector & u_new) { u_ = u_new; }

  /*!
   * \brief extract the pressure from the cfd solver
   */
  const InnerProdVector & get_press() { return cfd_.get_press(); }

  /*!
   * \brief uses the MDA state in u_ to update the cfd and csm states
   */
  void UpdateDisciplineStates();

  /*!
   * \brief defines the target pressure based on current pressure in cfd_
   */
  void CopyPressIntoTarget() { cfd_.set_press_targ(cfd_.get_press()); }

  /*!
   * \brief set the CFD and CSM initial condition for an NK MDA solve
   */
  void SetInitialCondition();

  /*!
   * \brief set the CFD and CSM initial condition for an NK MDA solve
   */
  void SetInitialConditionIntoVec(InnerProdVector & vec);

  /*!
   * \brief updates the discipline geometries based on the MDA nozzle object
   */
  void UpdateFromNozzle();

  /*!
   * \brief defines a sample Bspline-free MDA test problem
   */
  void InitializeTestProb();

  /*!
   * \brief tests grid-dependence on the aero-structural solution
   */
  void GridTest();

  /*!
   * \brief sets up the CFD solver for the MDA evaluation
   */
  void InitializeCFD(const InnerProdVector & x_coords,
                     const InnerProdVector & area);

  /*!
   * \brief sets up the CSM solver for the MDA evaluation
   */
  void InitializeCSM(const InnerProdVector & x_coords,
                     const InnerProdVector & y_coords,
                     const InnerProdVector & BCtype,
                     const InnerProdVector & BCval,
                     double E, double t, double w, double h);

  /*!
   * \brief calculate the system residual vector
   * \result residual based on u_ is calculated and stored in v_
   */
  void CalcResidual();

  /*!
   * \brief sets the equation scaling based on the residual res
   * \param[in] res - nonlinear MDA equation residual
   */
  void CalcRowScaling(const InnerProdVector & res);

  /*!
   * \brief applies scaling to the cfd and csm components of u
   * \param[in,out] u - vector to be scaled
   */
  void ScaleVector(InnerProdVector & u);

  void BuildAndFactorPreconditioner();

  /*!
   * \brief solves the aero-structural system with a Newton Krylow algorithm
   * \param[in] max_iter - maximum number of iterations permitted
   * \param[in] tol - tolerance with which to solve the system
   * \returns - total number of preconditioner calls
   */
  int NewtonKrylov(const int & max_iter, const double & tol, bool info=false);

  /*!
   * \brief solves for the linearized aero-structural problem
   * \param[in] max_iter - maximum number of iterations permitted
   * \param[in] tol - tolerance with which to solve the system
   * \param[in] rhs - the rhs of the linearized system
   * \param[out] sol - the solution vector
   * \returns total number of preconditioner calls
   */
  int SolveLinearized(const int & max_iter, const double & tol,
                      const InnerProdVector & rhs,
                      InnerProdVector & sol);

  /*!
   * \brief solves for the coupled adjoint variables using a Krylov solver
   * \param[in] max_iter - maximum number of iterations permitted
   * \param[in] tol - tolerance with which to solve the system
   * \param[in] dJdu - the rhs of the adjoint linear system
   * \param[out] psi - the adjoint solution vector
   * \returns total number of preconditioner calls
   */
  int SolveAdjoint(const int & max_iter, const double & tol,
                   const InnerProdVector & dJdu,
                   InnerProdVector & psi);

  /*!
   * \brief tests the AeroStructProduct using a finite-difference approximation
   * \pre the state defined in u_ must be "reasonable"
   */
  void TestMDAProduct();

  /*!
   * \brief tests the AeroStructTransposeProduct
   * \pre the state defined in u_ must be "reasonable"
   */
  void TestMDATransposedProduct();

  /*!
   * \brief prints out nodal displacements of the system design
   */
  void PrintDisplacements();

  /*!
   * \brief generates a .dat file for the solution, to be plotted with plot_nozzle.py
   * \pre must have solved, final nodal areas assigned to the CFD solver
   */
  void GetTecplot(const double & rho_ref, const double & a_ref,
                  const string & filename = "quasi1d.dat")
  { cfd_.WriteTecplot(rho_ref, a_ref, filename); }

// ======================================================================
// OPTIMIZATION ROUTINES
// ======================================================================

  /*!
   * \brief sets the number of Bspline design variables for the nozzle
   */
  void SetDesignVars(int num) { num_design_ = num; }

  /*!
   * \brief calculates (dR/dB)*vector product
   * \param[in] in - multiplied vector (num_design_)
   * \param[out] out - resultant vector (6*num_nodes_)
   */
  void Calc_dRdB_Product(InnerProdVector & in, InnerProdVector & out);

  /*!
   * \brief calculates (dR/dB)^T *vector product
   * \param[in] in - multiplied vector (6*num_nodes)
   * \param[out] out - resultant vector (num_design_)
   */
  void CalcTrans_dRdB_Product(InnerProdVector & in, InnerProdVector & out);

  /*!
   * \brief calculates (dS/dB)*vector product
   * \param[in] in - multiplied vector (num_design_)
   * \param[out] out - resultant vector (6*num_nodes_)
   */
  void Calc_dSdB_Product(InnerProdVector & in, InnerProdVector & out);

  /*!
   * \brief calculates (dS/dB)^T *vector product
   * \param[in] in - multiplied vector (6*num_nodes_)
   * \param[out] out - resultant vector (num_design_)
   */
  void CalcTrans_dSdB_Product(InnerProdVector & in, InnerProdVector & out);

  /*!
   * \brief calculates [dR/dB; dSdB]*vector product
   * \param[in] in - multiplied vector (num_design_)
   * \param[out] out - resultant vector (6*num_nodes_)
   */
  void AeroStructDesignProduct(InnerProdVector & in, InnerProdVector & out);

  /*!
   * \brief calculates [dR/dB; dSdB]^T *vector product
   * \param[in] in - multiplied vector (6*num_nodes_)
   * \param[out] out - resultant vector (num_design_)
   */
  void AeroStructDesignTransProduct(InnerProdVector & in, InnerProdVector & out);

  double CalcInverseDesign();

  void CalcInverseDesigndJdQ(InnerProdVector & dJdQ);

// ======================================================================

 private:
  Nozzle* nozzle_; ///< used to define problem and access Nozzle routines
  Quasi1DEuler cfd_; ///< used to access quasi_1d_euler matvec routines
  LECSM csm_; ///< used to access linear_elastic_csm routines
  double scale_cfd_; ///< used to scale linearized cfd equations
  double scale_csm_; ///< used to scale linearized csm equations
  double l_, h_, w_; ///< geometric domain length, height and width
  double E_, t_; ///< material properties
  InnerProdVector u_; ///< MDA solution vector
  InnerProdVector v_; ///< MDA residual vector
  int num_nodes_, order_, num_design_; ///< MDA discretization properties
  double p_ref_; ///< reference pressure for dimensionalization

  friend class AeroStructProduct;
  friend class AeroStructTransposeProduct;
  friend class AeroStructPrecond;
  friend class AeroStructTransposePrecond;
};

// ======================================================================

/*!
 * \class AeroStructProduct
 * \brief specialization of matrix-vector product for AeroStruct
 */
class AeroStructProduct:
    public kona::MatrixVectorProduct<InnerProdVector> {
public:

  AeroStructProduct(AeroStructMDA * mda) { mda_ = mda; }

  ~AeroStructProduct() {}

  void operator()(const InnerProdVector & u, InnerProdVector & v);

private:
  AeroStructMDA * mda_;
};

// ======================================================================

/*!
 * \class AeroStructTransposeProduct
 * \brief specialization of matrix-vector product for AeroStruct
 */
class AeroStructTransposeProduct:
    public kona::MatrixVectorProduct<InnerProdVector> {
public:

  AeroStructTransposeProduct(AeroStructMDA * mda) { mda_ = mda; }

  ~AeroStructTransposeProduct() {}

  void operator()(const InnerProdVector & u, InnerProdVector & v);

private:
  AeroStructMDA * mda_;
};

// ======================================================================

/*!
 * \class AeroStructPrecond
 * \brief specialization of matrix-vector product for AeroStruct
 */
class AeroStructPrecond:
    public kona::Preconditioner<InnerProdVector> {
public:

  AeroStructPrecond(AeroStructMDA * mda) { mda_ = mda; }

  ~AeroStructPrecond() {}

  void operator()(InnerProdVector & u, InnerProdVector & v);

private:
  AeroStructMDA * mda_;
};

// ======================================================================

/*!
 * \class AeroStructTransposePrecond
 * \brief specialization of matrix-vector product for AeroStruct
 */
class AeroStructTransposePrecond:
    public kona::Preconditioner<InnerProdVector> {
public:

  AeroStructTransposePrecond(AeroStructMDA * mda) { mda_ = mda; }

  ~AeroStructTransposePrecond() {}

  void operator()(InnerProdVector & u, InnerProdVector & v);

private:
  AeroStructMDA * mda_;
};
