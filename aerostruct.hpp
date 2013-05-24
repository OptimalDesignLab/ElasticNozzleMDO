/**
 * \file aerostruct.hpp
 * \brief header file for AeroStruct
 * \author Jason Hicken <jason.hicken@gmail.com>, Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

#pragma once

#include <math.h>

#include "./quasi_1d_euler/nozzle.hpp"
#include "./quasi_1d_euler/inner_prod_vector.hpp"
#include "./quasi_1d_euler/quasi_1d_euler.hpp"
#include "./linear_elastic_csm/lecsm.hpp"

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
  const InnerProdVector & get_u() const { return u_; }
  
  /*!
   * \brief define the MDA solution vector
   */
  void set_u(const InnerProdVector & u_new) { u_ = u_new; }
  
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
  void InitializeCFD(InnerProdVector & x_coords, InnerProdVector & area);

  /*!
   * \brief sets up the CSM solver for the MDA evaluation
   */
  void InitializeCSM(InnerProdVector & x_coords, InnerProdVector & y_coords,
                     InnerProdVector & BCtype, InnerProdVector & BCval,
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
  
  /*!
   * \brief solves the aero-structural system with a Newton Krylow algorithm
   * \param[in] max_iter - maximum number of iterations permitted
   * \param[in] tol - tolerance with which to solve the system
   * \returns - total number of preconditioner calls
   */
  int NewtonKrylov(const int & max_iter, const double & tol);

  /*!
   * \brief solves for the coupled adjoint variables using a Krylov solver
   * \param[in] max_iter - maximum number of iterations permitted
   * \param[in] tol - tolerance with which to solve the system
   * \param[in] dJdu - the rhs of the adjoint linear system
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
  void GetTecplot(const double & rho_ref, const double & a_ref)
  { cfd_.WriteTecplot(rho_ref, a_ref); }

// ======================================================================
// OPTIMIZATION ROUTINES
// ======================================================================

  /*!
   * \brief sets the number of Bspline design variables for the nozzle
   */
  void SetDesignVars(int num) { num_design_ = num; }

  /*!
   * \brief calculates (dR/dx)*vector product
   * \param[in] in - multiplied vector (num_design_)
   * \param[out] out - resultant vector (3*num_nodes_)
   */
  void Calc_dRdx_Product(InnerProdVector & in, InnerProdVector & out);

  /*!
   * \brief calculates (dR/dx)^T *vector product
   * \param[in] in - multiplied vector (3*num_nodes)
   * \param[out] out - resultant vector (num_design_)
   */
  void CalcTrans_dRdx_Product(InnerProdVector & in, InnerProdVector & out);

  /*!
   * \brief calculates (dS/dx)*vector product
   * \param[in] in - multiplied vector (num_design_)
   * \param[out] out - resultant vector (3*num_nodes_)
   */
  void Calc_dSdx_Product(InnerProdVector & in, InnerProdVector & out);

  /*!
   * \brief calculates (dS/dx)^T *vector product
   * \param[in] in - multiplied vector (3*num_nodes_)
   * \param[out] out - resultant vector (num_design_)
   */
  void CalcTrans_dSdx_Product(InnerProdVector & in, InnerProdVector & out);

  /*!
   * \brief calculates (dR/dx)*vector product
   * \param[in] in - multiplied vector (num_design_)
   * \param[out] out - resultant vector (3*num_nodes_)
   */
  void AeroStructDesignProduct(InnerProdVector & in, InnerProdVector & out);

// ======================================================================

 private:
  Nozzle* nozzle_; ///< used to define problem and access Nozzle routines
  Quasi1DEuler cfd_; ///< used to access quasi_1d_euler matvec routines
  LECSM csm_; ///< used to access linear_elastic_csm routines
  double scale_cfd_; ///< used to scale linearized cfd equations
  double scale_csm_; ///< used to scale linearized csm equations
  InnerProdVector u_;
  InnerProdVector v_;
  int num_nodes_, order_, num_design_;
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
