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
      csm_(num_nodes)
  { num_nodes_ = num_nodes;
    order_ = order; }

  ~AeroStructMDA() {} ///< class destructor

  void InitializeTestProb();

  /*!
   * \brief returns a pointer to the CFD discipline solver
   * \return cfd_ member value
   */
  Quasi1DEuler get_cfd() { return * cfd_;}

  /*!
   * \brief returns a pointer to the CSM discipline solver
   * \return csm_ member value
   */
  LECSM get_csm() { return * csm_;}

  /*!
   * \brief calculate the system residual vector
   * \result residual based on u_ is calculated and stored in v_
   */
  void CalcResidual();

  /*!
   * \brief returns the L2 norm of the system residual
   * \return v_.Norm2() value
   */
  double ResidualNorm() { return v_.Norm2(); };

  /*!
   * \brief solves the aero-structural system with a Newton Krylow algorithm
   * \param[in] max_iter - maximum number of iterations permitted
   * \param[in] tol - tolerance with which to solve the system
   * \returns - total number of preconditioner calls
   */
  void NewtonKrylov(const int & max_iter, const double & tol);

  void GetTecplot(const double & rho_ref, const double & a_ref)
  { cfd_->WriteTecplot(rho_ref, a_ref); }

 private:
  Quasi1DEuler * cfd_; ///< used to access quasi_1d_euler matvec routines
  LECSM * csm_; ///< used to access linear_elastic_csm routines
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
    public kona::MatrixVectorProduct<InnerProdVector> {
public:

  AeroStructPrecond(AeroStructMDA * mda) { mda_ = mda; }

  ~AeroStructPrecond() {}

  void operator()(InnerProdVector & u, InnerProdVector & v);

private:
  AeroStructMDA * mda_;
}; 
