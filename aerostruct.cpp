/**
 * \file aerostruct.cpp
 * \brief function defintions for aerostruct member functions
 * \author Jason Hicken <jason.hicken@gmail.com>, Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

#include "./aerostruct.hpp"

#include <math.h>

#include <ostream>
#include <iostream>
#include <fstream>
#include <string>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <krylov.hpp>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::ofstream;

// ======================================================================

void AeroStructMDA::CalcResidual()
{
  // Split system u into CSM and CFD vectors
  num_nodes = cfd_->get_num_nodes();  
  InnerProdVector u_cfd(3*num_nodes, 0.0), v_cfd(3*num_nodes, 0.0),
      u_csm(3*num_nodes, 0.0), v_csm(3*num_nodes, 0.0);
  for (int i = 0; i < 3*num_nodes; i++) {
    u_cfd(i) = u(i);
    u_csm(i) = u(3*num_nodes+i);
  }
  // Update the discipline vectors
  cfd_->set_q(u_cfd);                 // set the flow variables
  csm_->set_u(u_csm);                 // set the nodal displacements

  // CFD Operations
  csm_->CalcStateVars();              // calculate the area and x coords
  cfd_->set_area(csm_->get_area());   // set the area
  cfd_->set_x_coord(csm_->get_x());   // set the nodal x coordinates
  cfd_->CalcResidual();               // calculate the CFD residual
  
  // CSM Operations
  csm_->set_press(cfd_->get_press()); // set the pressures from CFD
  csm_->CalcResidual();               // calculate the CSM residual

  // Retreive the discipline residuals
  v_cfd = cfd_->get_res();
  v_csm = csm_->get_res();

  // Merge the disciplines into the system residual
  for (int i=0; i<3*num_nodes; i++) {
    v_(i) = v_cfd(i);
    v_(3*num_nodes+i) = v_csm(i);
  }
}

// ======================================================================

void AeroStructMDA::NewtonKrylov(const int & max_iter, const double & tol)
{
  kona::MatrixVectorProduct<InnerProdVector>* 
      mat_vec = new AeroStructProduct(this);
  kona::Preconditioner<InnerProdVector>*
      precond = new BuildPrecondtioner(this);

  string filename = "aero_struct_primal.dat";
  ofstream fout(filename.c_str());

  int iter = 0;
  int precond_calls = 0;
  while (iter < max_iter) {    
    // evaluate the residual and its norm
    CalcResidual();  // merge aero residual with struct residual
    double norm = ResidualNorm();   // evaluate the L2 norm
    InnerProdVector b(-v_);
    cout << "iter = " << iter
         << ": L2 norm of residual = " << norm << endl;
    if ( (norm < tol) || (norm < 1.e-14) ) {
      cout << "AeroStructProduct: NewtonKrylov converged" << endl;
      return precond_calls;
    }

    // re-calculate the CFD preconditioner at each iteration
    cfd_->BuildAndFactorPreconditioner();

    // solve for the Newton update du and add to u
    int m = 10;
    double tol = 1.0e-2;
    InnerProdVector dq(3*num_nodes_, 0.0);
    int krylov_precond_calls;
    try {
      kona::FGMRES(m, tol, b, du, *mat_vec, *precond,
                   krylov_precond_calls, fout);
    } catch (...) {
      cout << "AeroStructProduct: FGMRES failed in NewtonKrylov" << endl;
      return -precond_calls;
    }
      
    u_ += du;
    precond_calls += krylov_precond_calls;
    iter++;
  }
  // if we get here, we failed to converge
  cout << "AeroStructProduct::NewtonKrylov(): "
       << "failed to converge in " << max_iter << " iterations." << endl;
  //throw(-1);
  return -precond_calls;
}

// ======================================================================

void AeroStructProduct::operator()(const InnerProdVector & u, 
                                   InnerProdVector & v) {
  // decompose u into its cfd and csm parts, and create some work arrays
  num_nodes = cfd_->get_num_nodes();  
  InnerProdVector u_cfd(3*num_nodes, 0.0), v_cfd(3*num_nodes, 0.0),
      u_csm(3*num_nodes, 0.0), v_csm(3*num_nodes, 0.0),
      wrk(num_nodes, 0.0);
  for (int i = 0; i < 3*num_nodes; i++) {
    u_cfd(i) = u(i);
    u_csm(i) = u(3*num_nodes+i);
  }

  // Denote the Aerostructural Jacobian-vector product by
  //    | A  B | | u_cfd | = | v_cfd |
  //    | C  D | | u_csm |   | v_csm |

  // Compute A*u_cfd
  cfd_->JacobianStateProduct(u_cfd, v_cfd);

  // Compute D*u_csm
  csm_->Calc_dSdu_Product(u_csm, v_csm);

  // Compute B*u_csm = (dR/dA)*(dA/d(delA))*(d(delA)/du)*u_csm =
  csm_->Calc_dAdu_Product(u_csm, wrk);
  // NOTE: below, I assume u_csm is not needed anymore, so I can use it for work
  cfd_->JacobianAreaProduct(wrk, u_csm);
  v_cfd += u_csm;
  
  // Compute C*u_cfd = (dS/dp)*(dp/dq)*u_cfd = (dS/dp)*wrk
  cfd_->CalcDPressDQProduct(u_cfd, wrk);
  // NOTE: below, I assume u_cfd is not needed anymore so I can use it for work (payback!)
  csm_->Calc_dSdp_Product(wrk, u_cfd);
  v_csm += u_cfd;

  // Finally, assemble v from its cfd and csm parts
  for (int i = 0; i < 3*num_nodes; i++) {
    v(i) = v_cfd(i);
    v(3*num_nodes+i) = v_csm(i);
  }
}

// ======================================================================

void AeroStructPrecond::operator()(InnerProdVector & u, InnerProdVector & v)
{
  // inherit the preconditioner calculated at every iteration for the CFD
  kona::Preconditioner<InnerProdVector>*
      precond_cfd = new cfd_->ApproxJacobian(cfd_);

  // build identity matrix static preconditioner for the structural solver
  kona::Preconditioner<InnerProdVector>*
      precond_csm(3*num_nodes)

  // merge the preconditioners and pass it up
}