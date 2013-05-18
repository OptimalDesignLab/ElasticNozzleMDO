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

void AeroStructMDA::InitializeTestProb()
{
  // start defining the nozzle
  double length = 10.0;
  InnerProdVector x_coord(num_nodes_, 0.0);
  InnerProdVector y_coord(num_nodes_, 0.0);
  InnerProdVector area(num_nodes_, 0.0);
  for (int i = 0; i < num_nodes_; i++) {
    // evenly spaced nodes along the x
    x_coord(i) = i*length/(num_nodes_-1);
    // parabolic nozzle wall for y coords
    y_coord(i) = 0.01*(length-x_coord(i))*x_coord(i);
    // area based on a 1-by-1 square intake cross-section
    area(i) = 2*(1-y_coord(i)); 
  }

  // set the CFD "mesh"
  cfd_.set_x_coord(x_coord);
  cfd_.set_area(area);

  // define and set the left-end boundary conditions
  double rho_ref = 1.14091202011454;
  double p = 9.753431315656936E4;
  double rho = rho_ref;
  double rho_u = rho*65.45103620864865;
  double a_ref = sqrt(kGamma*p/rho);
  double e = p/(kGamma-1.0) + 0.5*rho_u*rho_u/rho;
  // nondimensionalize
  rho_u /= (a_ref*rho_ref);
  e /= (rho_ref*a_ref*a_ref);
  rho = 1.0; // i.e. non-dimensionalize density by itself
  cfd_.set_bc_left(rho, rho_u, e);

  // set the initial flow to the left boundary
  cfd_.InitialCondition(rho, rho_u, e);

  // define and set the right-end boundary conditions
  p = 9.277211161772544E4;
  rho = 1.10083845647356;
  rho_u = rho * 113.0560581652928;
  e = p/(kGamma-1.0) + 0.5*rho_u*rho_u/rho;
  // nondimensionalize
  rho /= rho_ref;
  rho_u /= (rho_ref*a_ref);
  e /= (rho_ref*a_ref*a_ref);
  cfd_.set_bc_right(rho, rho_u, e);

  // define any discretization and solver parameters
  cfd_.set_diss_coeff(0.04);

  // generate CSM mesh
  csm_.GenerateMesh(x_coord, y_coord);

  // determine the nodal structural boundary conditions
  InnerProdVector BCtype(3*num_nodes_, 0.0);
  InnerProdVector BCval(3*num_nodes_, 0.0);
  for (int i=0; i<num_nodes_; i++) {
    BCtype(3*i) = 0;
    BCtype(3*i+1) = -1;
    BCtype(3*i+2) = -1;
    BCval(3*i) = 0;
    BCval(3*i+1) = 0;
    BCval(3*i+2) = 0;
  }
  BCtype(0) = 0;
  BCtype(1) = 0;
  BCtype(2) = -1;
  BCtype(3*num_nodes_-3) = 0;
  BCtype(3*num_nodes_-2) = 0;
  BCtype(3*num_nodes_-1) = -1;
  csm_.SetBoundaryConds(BCtype, BCval);

  // set material properties for CSM
  double E = 100000000;   // Young's modulus
  double w = 1;           // fixed width of nozzle
  double t = 0.03;        // fixed beam element thickness
  double h = 1;           // max height of the nozzle
  csm_.set_material(E, t, w, h);

  csm_.InspectMesh();
}

// ======================================================================

void AeroStructMDA::CalcResidual()
{
  // Split system u into CSM and CFD vectors  
  InnerProdVector u_cfd(3*num_nodes_, 0.0), u_csm(3*num_nodes_, 0.0);
  for (int i = 0; i < 3*num_nodes_; i++) {
    u_cfd(i) = u_(i);
    u_csm(i) = u_(3*num_nodes_+i);
  }
  // Update the discipline vectors
  cfd_.set_q(u_cfd);                 // set the flow variables
  csm_.set_u(u_csm);                 // set the nodal displacements

  // CFD Operations
  csm_.CalcStateVars();              // calculate the area and x coords
  cfd_.set_area(csm_.get_area());   // set the area
  cfd_.set_x_coord(csm_.get_x());   // set the nodal x coordinates
  cfd_.CalcResidual();               // calculate the CFD residual
  
  // CSM Operations
  csm_.set_press(cfd_.get_press()); // set the pressures from CFD
  csm_.CalcResidual();               // calculate the CSM residual

  // Retreive the discipline residuals
  const InnerProdVector & v_cfd = cfd_.get_res();
  const InnerProdVector & v_csm = csm_.get_res();

  // Merge the disciplines into the system residual
  for (int i=0; i<3*num_nodes_; i++) {
    v_(i) = v_cfd(i);
    v_(3*num_nodes_+i) = v_csm(i);
  }
}

// ======================================================================

int AeroStructMDA::NewtonKrylov(const int & max_iter, const double & tol)
{
  kona::MatrixVectorProduct<InnerProdVector>* 
      mat_vec = new AeroStructProduct(this);
  kona::Preconditioner<InnerProdVector>*
      precond = new AeroStructPrecond(this);

  string filename = "aero_struct_primal.dat";
  ofstream fout(filename.c_str());

  int iter = 0;
  int precond_calls = 0;
  while (iter < max_iter) {    
    // evaluate the residual and its norm
    CalcResidual();  // merge aero residual with struct residual
    double norm = v_.Norm2();   // evaluate the L2 norm
    cout << "iter = " << iter
         << ": L2 norm of residual = " << norm << endl;
    if ( (norm < tol) || (norm < 1.e-14) ) {
      cout << "Solver: NewtonKrylov converged!" << endl;
      return precond_calls;
    }

    // solve for the Newton update du and add to u
    int m = 10;
    double tol = 1.0e-2;
    InnerProdVector du(6*num_nodes_, 0.0);
    int krylov_precond_calls;
    try {
      kona::FGMRES(m, tol, v_, du, *mat_vec, *precond,
                   krylov_precond_calls, fout);
    } catch (...) {
      cout << "Solver: FGMRES failed in NewtonKrylov!" << endl;
      return -precond_calls;
    }
      
    u_ += du;
    precond_calls += krylov_precond_calls;
    iter++;
  }
  // if we get here, we failed to converge
  cout << "AeroStructMDA::NewtonKrylov(): "
       << "failed to converge in " << max_iter << " iterations." << endl;
  //throw(-1);
  return -precond_calls;
}

// ======================================================================

void AeroStructProduct::operator()(const InnerProdVector & u, 
                                   InnerProdVector & v) 
{
  // decompose u into its cfd and csm parts, and create some work arrays
  int nnp = mda_->num_nodes_;
  InnerProdVector u_cfd(3*nnp, 0.0), v_cfd(3*nnp, 0.0),
      u_csm(3*nnp, 0.0), v_csm(3*nnp, 0.0),
      wrk(nnp, 0.0);
  for (int i = 0; i < 3*nnp; i++) {
    u_cfd(i) = u(i);
    u_csm(i) = u(3*nnp+i);
  }

  // Denote the Aerostructural Jacobian-vector product by
  //    | A  B | | u_cfd | = | v_cfd |
  //    | C  D | | u_csm |   | v_csm |

  // Compute A*u_cfd
  mda_->cfd_.JacobianStateProduct(u_cfd, v_cfd);

  // Compute D*u_csm
  mda_->csm_.Calc_dSdu_Product(u_csm, v_csm);

  // Compute B*u_csm = (dR/dA)*(dA/d(delA))*(d(delA)/du)*u_csm =
  mda_->csm_.Calc_dAdu_Product(u_csm, wrk);
  // NOTE: below, I assume u_csm is not needed anymore, so I can use it for work
  mda_->cfd_.JacobianAreaProduct(wrk, u_csm);
  v_cfd += u_csm;
  
  // Compute C*u_cfd = (dS/dp)*(dp/dq)*u_cfd = (dS/dp)*wrk
  mda_->cfd_.CalcDPressDQProduct(u_cfd, wrk);
  // NOTE: below, I assume u_cfd is not needed anymore so I can use it for work (payback!)
  mda_->csm_.Calc_dSdp_Product(wrk, u_cfd);
  v_csm += u_cfd;

  // Finally, assemble v from its cfd and csm parts
  for (int i = 0; i < 3*nnp; i++) {
    v(i) = v_cfd(i);
    v(3*nnp+i) = v_csm(i);
  }
}

// ======================================================================

void AeroStructPrecond::operator()(InnerProdVector & u, InnerProdVector & v)
{
  // decompose u into its cfd and csm parts, and create some work arrays
  int nnp = mda_->num_nodes_;
  InnerProdVector u_cfd(3*nnp, 0.0), v_cfd(3*nnp, 0.0),
      v_csm(3*nnp, 0.0);
  for (int i = 0; i < 3*nnp; i++) {
    u_cfd(i) = u(i);
    v_csm(i) = u(3*nnp+i); // v_csm = u_csm (no preconditioning for CSM)
  }
  // inherit the preconditioner calculated at every iteration for the CFD
  mda_->cfd_.BuildAndFactorPreconditioner();
  mda_->cfd_.Precondition(u_cfd, v_cfd);

  // merge the preconditioners and pass it up
  for (int i = 0; i < 3*nnp; i++) {
    v(i) = v_cfd(i);
    v(3*nnp+i) = v_csm(i);
  }

}