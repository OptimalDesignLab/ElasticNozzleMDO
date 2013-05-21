/**
 * \file aerostruct.hpp
 * \brief header file for AeroStruct
 * \author Jason Hicken <jason.hicken@gmail.com>, Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

#pragma once

#include <math.h>

#include "./quasi_1d_euler/inner_prod_vector.hpp"
#include "./quasi_1d_euler/quasi_1d_euler.hpp"
#include "./linear_elastic_csm/lecsm.hpp"

// forward declarations
class AeroStructProduct;
class AeroStructPrecond;

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

  ~AeroStructMDA() {} ///< class destructor

  void InitializeTestProb();

  /*!
   * \brief calculate the system residual vector
   * \result residual based on u_ is calculated and stored in v_
   */
  void CalcResidual();

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
   * \brief tests the AeroStructProduct using a finite-difference approximation
   *
   * \pre the state defined in u_ must be "reasonable"
   */
  void TestMDAProduct();

  void GetTecplot(const double & rho_ref, const double & a_ref)
  { cfd_.WriteTecplot(rho_ref, a_ref); }

 private:
  Quasi1DEuler cfd_; ///< used to access quasi_1d_euler matvec routines
  LECSM csm_; ///< used to access linear_elastic_csm routines
  double scale_cfd_; ///< used to scale linearized cfd equations
  double scale_csm_; ///< used to scale linearized csm equations
  InnerProdVector u_;
  InnerProdVector v_;
  int num_nodes_, order_;

  friend class AeroStructProduct;
  friend class AeroStructPrecond;
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
