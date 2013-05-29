/**
 * \file aerostruct.cpp
 * \brief function defintions for aerostruct member functions
 * \author Jason Hicken <jason.hicken@gmail.com>, Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

#include "./aerostruct.hpp"
#include "./constants.hpp"

#include <math.h>

#include <ostream>
#include <iostream>
#include <fstream>
#include <string>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <krylov.hpp>

#include "./quasi_1d_euler/exact_solution.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::ofstream;

// solution parameters
//static const double kAreaStar = 0.8; // for the exact solution
//static const double kTempStag = 300.0;
//static const double kPressStag = 100000;
//static const double kRGas = 287.0;

// ======================================================================

void AeroStructMDA::UpdateFromNozzle()
{
  // evenly spaced x-coordinates along the length of nozzle
  InnerProdVector x_coord(num_nodes_, 0.0);
  for (int i = 0; i < num_nodes_; i++)
    x_coord(i) = i*l_/(num_nodes_-1);

  // query the nozzle object for nodal areas
  InnerProdVector area = nozzle_->Area(x_coord);

  // reverse-calculate y-coordinates from the nodal areas
  InnerProdVector y_coord(num_nodes_, 0.0);
  for (int i = 0; i < num_nodes_; i++)
    y_coord(i) = (h_/2) - (area(i)/(2*w_));

  // update solver properties
  cfd_.set_area(area);
  csm_.set_coords(x_coord, y_coord);
  csm_.UpdateMesh();
}

// ======================================================================

void AeroStructMDA::InitializeTestProb()
{
  // set material properties for CSM
  double E = 100000000;   // Young's modulus
  double w = 1.0;           // fixed width of nozzle
  double t = 0.01;        // fixed beam element thickness
  double h = 2;           // max height of the nozzle

  // start defining the nozzle
  double length = 1.0;
  InnerProdVector x_coord(num_nodes_, 0.0);
  InnerProdVector y_coord(num_nodes_, 0.0);
  InnerProdVector area(num_nodes_, 0.0);
  double a = area_left;
  double b = 4.0*area_mid - 5.0*area_left + area_right;
  double c = -4.0*(area_right -2.0*area_left + area_mid);
  double d = 4.0*(area_right - area_left);
  for (int i = 0; i < num_nodes_; i++) {
    // evenly spaced nodes along the x
    x_coord(i) = i*length/(num_nodes_-1);
    //y_coord(i) = 0.0025*(10 - x_coord(i))*x_coord(i);
    //y_coord(i) = 0.25*(1.0 - x_coord(i))*x_coord(i);
    //area(i) = w*(h - 2*y_coord(i));
    //area(i) = area_left + (area_right - area_left)*x_coord(i)/length;
    area(i) = a + x_coord(i)*(b + x_coord(i)*(c + x_coord(i)*d));    
    y_coord(i) = 0.5*(h - area(i)/width);
  }

  // initialize the CFD solver
  InitializeCFD(x_coord, area);

  // determine the nodal structural boundary conditions
  InnerProdVector BCtype(3*num_nodes_, 0.0);
  InnerProdVector BCval(3*num_nodes_, 0.0);
  for (int i=0; i<num_nodes_; i++) {
    BCtype(3*i) = 0; //-1;
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

  // initialize the CSM solver
  InitializeCSM(x_coord, y_coord, BCtype, BCval, E, t, w, h);

  csm_.InspectMesh();
}

// ======================================================================

void AeroStructMDA::GridTest() {

  //CalcResidual();
  
  int fine_nodes = 2*(num_nodes_-1) + 1;
  InnerProdVector x_coord(num_nodes_, 0.0), x_fine(fine_nodes, 0.0);
  InnerProdVector y_coord(num_nodes_, 0.0), y_fine(fine_nodes, 0.0);  
  InnerProdVector area(num_nodes_, 0.0), area_fine(fine_nodes, 0.0);
  InnerProdVector press(num_nodes_, 0.0), press_fine(fine_nodes, 0.0);

#if 0
  InnerProdVector u_csm(3*num_nodes_, 0.0);
  for (int i = 0; i < 3*num_nodes_; i++) {
    u_csm(i) = 0.0;
  }
  double length = 1.0;
  for (int i = 0; i < num_nodes_; i++) {
    // evenly spaced nodes along the x
    x_coord(i) = i*length/(num_nodes_-1);
    y_coord(i) = 0.25*(1.0 - x_coord(i))*x_coord(i);    
  }

  csm_.set_u(u_csm);
  csm_.set_press(cfd_.get_press()); // set the pressures from CFD
  csm_.CalcResidual();
  csm_.Solve();
  csm_.set_u(csm_.get_u());
  csm_.CalcCoordsAndArea();
  cfd_.set_area(csm_.get_area());
  cfd_.WriteTecplot(1.0, 1.0, "csm_area.dat");
#endif

  // interpolate area, x, and y from csm, and press from cfd
  csm_.CalcCoordsAndArea();
  area = csm_.get_area();
  x_coord = csm_.get_x();
  y_coord = csm_.get_y();
  press = cfd_.get_press();
  press *= p_ref_;
  for (int i = 0; i < num_nodes_; i++) {
    area_fine(2*i) = area(i);
    x_fine(2*i) = x_coord(i);
    press_fine(2*i) = press(i) - kPressStag;
    if (i < num_nodes_-1) {
      area_fine(2*i+1) = 0.5*(area(i) + area(i+1));
      x_fine(2*i+1) = 0.5*(x_coord(i) + x_coord(i+1));
      press_fine(2*i+1) = 0.5*(press(i) + press(i+1)) - kPressStag;
    }
  }

  // rerun cfd on refined grid and interpolated area
  cfd_.ResizeGrid(x_fine);
  num_nodes_ = fine_nodes;
  InitializeCFD(x_fine, area_fine);
  //cfd_.set_area(area_fine);
  cfd_.NewtonKrylov(30, 1e-8);
  cfd_.WriteTecplot(1.0, 1.0, "refined_quasi1d.dat");

  // create a CSM and run with interpolated pressure
  LECSM csm_fine(fine_nodes);
  double E = 100000000;   // Young's modulus
  double w = 1.0;           // fixed width of nozzle
  double t = 0.01;        // fixed beam element thickness
  double h = 2;           // max height of the nozzle

  // determine the nodal structural boundary conditions
  InnerProdVector BCtype(3*fine_nodes, 0.0);
  InnerProdVector BCval(3*fine_nodes, 0.0);
  for (int i=0; i<fine_nodes; i++) {
    BCtype(3*i) = 0; //-1;
    BCtype(3*i+1) = -1;
    BCtype(3*i+2) = -1;
    BCval(3*i) = 0;
    BCval(3*i+1) = 0;
    BCval(3*i+2) = 0;
  }
  BCtype(0) = 0;
  BCtype(1) = 0;
  BCtype(2) = -1;
  BCtype(3*fine_nodes-3) = 0;
  BCtype(3*fine_nodes-2) = 0;
  BCtype(3*fine_nodes-1) = -1;

  double length = 1.0;
  for (int i = 0; i < fine_nodes; i++) {
    x_fine(i) = i*length/(fine_nodes-1);
    y_fine(i) = 0.25*(1.0 - x_fine(i))*x_fine(i);
  }
  
  // set material properties
  csm_fine.set_material(E, t, w, h);
  // create the CSM mesh
  csm_fine.GenerateMesh(x_fine, y_fine);
  // set the nodal degrees of freedom
  csm_fine.SetBoundaryConds(BCtype, BCval);

  csm_fine.set_press(press_fine); // alternatively use cfd_.get_press() 
  csm_fine.Solve();
  //csm_fine.set_u(csm_fine.get_u());  // this is redundant now (for this test)
  csm_fine.CalcCoordsAndArea();
  
  cfd_.set_area(csm_fine.get_area());
  cfd_.WriteTecplot(1.0, 1.0, "refined_csm_area.dat");


}

// ======================================================================

void AeroStructMDA::InitializeCFD(const InnerProdVector & x_coord,
                                  const InnerProdVector & area)
{
  // set the CFD "mesh"
  cfd_.set_x_coord(x_coord);
  cfd_.set_area(area);

  // define reference and boundary conditions
  double area_left = area(0);
  double area_right = area(num_nodes_-1);
  double rho, rho_u, e;
  CalcFlowExact(kGamma, kRGas, kAreaStar, area_left, true, 
                kTempStag, kPressStag, rho, rho_u, e);
  double rho_ref = rho;
  double press = (kGamma - 1.0)*(e - 0.5*rho_u*rho_u/rho);
  p_ref_ = press;
  double a_ref = sqrt(kGamma*press/rho_ref);
  double rho_L = 1.0;
  double rho_u_L = rho_u/(a_ref*rho_ref);
  double e_L = e/(rho_ref*a_ref*a_ref);
  CalcFlowExact(kGamma, kRGas, kAreaStar, area_right, true,
                kTempStag, kPressStag, rho, rho_u, e);
  double rho_R = rho/rho_ref;
  double rho_u_R = rho_u/(a_ref*rho_ref);
  double e_R =  e/(rho_ref*a_ref*a_ref); 

  // set boundary and initial conditions
  cfd_.set_bc_left(rho_L, rho_u_L, e_L);
  cfd_.InitialCondition(rho_R, rho_u_R, e_R);
  cfd_.set_bc_right(rho_R, rho_u_R, e_R);
  
  // define any discretization and solver parameters
  cfd_.set_diss_coeff(0.04);

  // initialize the CFD part of the aerostructural solution guess
  for (int i = 0; i < num_nodes_; i++) {
    u_(3*i) = rho_R;
    u_(3*i+1) = rho_u_R;
    u_(3*i+2) = e_R;
  }
}

// ======================================================================

void AeroStructMDA::InitializeCSM(const InnerProdVector & x_coord,
                                  const InnerProdVector & y_coord,
                                  const InnerProdVector & BCtype,
                                  const InnerProdVector & BCval,
                                  double E, double t, double w, double h)
{
  // set material properties
  csm_.set_material(E, t, w, h);
  // create the CSM mesh
  csm_.GenerateMesh(x_coord, y_coord);
  // set the nodal degrees of freedom
  csm_.SetBoundaryConds(BCtype, BCval);
  // initialize the CSM part of the aerostructural solution guess
  for (int i = 0; i < num_nodes_; i++) {
    u_(3*(num_nodes_+i)) = 0.0;
    u_(3*(num_nodes_+i)+1) = 0.0;
    u_(3*(num_nodes_+i)+2) = 0.0;
  }
}
// ======================================================================

void AeroStructMDA::CalcResidual()
{ 
  // Reset CSM coordinates back to the original geometry
  //csm_.ResetCoords();
  
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
  csm_.CalcCoordsAndArea();         // calculate the displaced x,y and area  
  cfd_.set_area(csm_.get_area());   // set the area
  cfd_.set_x_coord(csm_.get_x());   // set the nodal x coordinates
  cfd_.CalcResidual();              // calculate the CFD residual

  // CSM Operations
  InnerProdVector press(num_nodes_, 0.0), press_stag(num_nodes_, kPressStag);
  press = cfd_.get_press();
  press *= p_ref_;
  press -= press_stag;  // because pressure on other side of nozzle
  csm_.set_press(press);
  csm_.CalcResidual();               // calculate the CSM residual

  //csm_.ResetCoords();
  
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

void AeroStructMDA::CalcRowScaling(const InnerProdVector & res) {
  scale_cfd_ = 0.0;
  scale_csm_ = 0.0;
  for (int i=0; i<3*num_nodes_; i++) {
    scale_cfd_ += res(i)*res(i);
    scale_csm_ += res(3*num_nodes_+i)*res(3*num_nodes_+i);
  }
  scale_cfd_ = 1.0/sqrt(scale_cfd_);
  scale_csm_ = 1.0/sqrt(scale_csm_);
}

// ======================================================================

void AeroStructMDA::ScaleVector(InnerProdVector & u) {
  for (int i=0; i<3*num_nodes_; i++) {
    u(i) *= scale_cfd_;
    u(3*num_nodes_+i) *= scale_csm_;
  }
}

// ======================================================================

int AeroStructMDA::NewtonKrylov(const int & max_iter, const double & tol)
{
  kona::MatrixVectorProduct<InnerProdVector>* 
      mat_vec = new AeroStructProduct(this);
  kona::Preconditioner<InnerProdVector>*
      precond = new AeroStructPrecond(this);

  string filename = "mda_primal_krylov.dat";
  ofstream fout(filename.c_str());

#if 0
  cfd_.NewtonKrylov(max_iter, tol);
  cfd_.WriteTecplot(1.0, 1.0, "undeformed_flow.dat");
#endif
  
  int iter = 0;
  int precond_calls = 0;
  double norm0;
  while (iter < max_iter) {
    // evaluate the residual and its norm
    CalcResidual();  // merge aero residual with struct residual
    InnerProdVector b(-v_);
    double norm = b.Norm2();   // evaluate the L2 norm
    if (iter == 0) norm0 = norm;
    cout << "iter = " << iter
         << ": L2 norm of residual = " << norm << endl;
    if ( (norm < tol*norm0) || (norm < 1.e-14) ) {
      cout << "Solver: NewtonKrylov converged!" << endl;
      return precond_calls;
    }
    
#if 0
    // reset CSM grid to original geometry before peforming JacobianVectorProduct
    csm_.ResetCoords();
    csm_.CalcCoordsAndArea();
    cfd_.set_area(csm_.get_area());
    cfd_.set_x_coord(csm_.get_x());
#endif

    // scale residual
    CalcRowScaling(b);
    ScaleVector(b);
    
    // Update CFD preconditioner
    cfd_.BuildAndFactorPreconditioner();
    
    // solve for the Newton update du and add to u
    int m = 100;
    double krylov_tol = std::min(0.1, norm/norm0);
    krylov_tol = std::max(krylov_tol, tol/norm);
    InnerProdVector du(6*num_nodes_, 0.0);
    int krylov_precond_calls;
    try {
      kona::FGMRES(m, krylov_tol, b, du, *mat_vec, *precond,
                   krylov_precond_calls, fout);
    } catch (...) {
      cout << "Solver: FGMRES failed in NewtonKrylov!" << endl;
      return -precond_calls;
    }

    // update the individual discipline states
    double damp = 1.00;
    u_ += damp*du;    

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

int AeroStructMDA::SolveAdjoint(const int & max_iter, const double & tol,
                                const InnerProdVector & dJdu,
                                InnerProdVector & psi) {
  kona::MatrixVectorProduct<InnerProdVector>* 
      mat_vec = new AeroStructTransposeProduct(this);
  kona::Preconditioner<InnerProdVector>*
      precond = new AeroStructTransposePrecond(this);
  
  string filename = "mda_adjoint_krylov.dat";
  ofstream fout(filename.c_str());

  // Update CFD preconditioner
  cfd_.BuildAndFactorPreconditioner();
  
  psi = 0.0;
  int precond_calls = 0;
  kona::FGMRES(max_iter, tol, dJdu, psi, *mat_vec, *precond,
               precond_calls, fout);
  //ScaleVector(psi);
  fout.close();
  return precond_calls;
}

// ======================================================================

void AeroStructMDA::TestMDAProduct()
{
  // create a random vector to apply Jacobian to
  InnerProdVector u(6*num_nodes_, 0.0), v(6*num_nodes_, 0.0), 
      v_fd(6*num_nodes_, 0.0), u_save(6*num_nodes_, 0.0);
  boost::random::mt19937 gen;
  boost::random::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int i = 0; i < 6*num_nodes_; i++)
    u(i) = dist(gen);

  // set the row scaling
  scale_cfd_ = 1.0;
  scale_csm_ = 1.0;
  
  u_save = u_;  // save the state for later
  double ref_save = p_ref_;
  p_ref_ = 1.0;
  
  kona::MatrixVectorProduct<InnerProdVector>* 
      mat_vec = new AeroStructProduct(this);  
  (*mat_vec)(u, v);
  delete mat_vec;
  
  // evaluate residual and save
  CalcResidual();
  v_fd = v_;
  
  // perturb flow and re-evaluate residual
  double fd_eps = 1.E-5;
  u_ += fd_eps*u;
  CalcResidual();
  v_fd -= v_;
  v_fd /= -fd_eps; // minus sign accounts for switch in order

  // take difference between two products and store in q_ for output
  u.EqualsAXPlusBY(1.0, v, -1.0, v_fd);

#if 0
  // uncomment to print error at each variable
  cout << "TestMDAProduct: product elements corresponding to cfd:" << endl;
  for (int i = 0; i < 3*num_nodes_; i++)
    cout << "delta v(" << i << ") = " << u(i) << endl;
  cout << "TestMDAProduct: product elements corresponding to csm:" << endl;
  for (int i = 3*num_nodes_; i < 6*num_nodes_; i++) {
    //cout << "v(" << i << ")    = " << v(i) << endl;
    //cout << "v_fd(" << i << ") = " << v_fd(i) << endl;
    cout << "delta v(" << i << ") = " << u(i) << endl;
  }
#endif
  
  double L2_error = u.Norm2();
  cout << "TestMDAProduct: "
       << "L2 error between analytical and FD Jacobian-vector product: "
       << L2_error << endl;

  // reset the state
  u_ = u_save;
  p_ref_ = ref_save;
}

// ======================================================================

void AeroStructMDA::TestMDATransposedProduct()
{
  // create a random vector to apply Jacobian to
  InnerProdVector u(6*num_nodes_, 0.0), v(6*num_nodes_, 0.0), 
      w(6*num_nodes_, 0.0);
  boost::random::mt19937 gen;
  boost::random::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (int i = 0; i < 6*num_nodes_; i++) {
    u(i) = dist(gen);
    v(i) = dist(gen);
  }
  
  double ref_save = p_ref_;
  p_ref_ = 1.0;
  
  // set the row scaling
  scale_cfd_ = 1.0;
  scale_csm_ = 1.0;
  
  kona::MatrixVectorProduct<InnerProdVector>* 
      mat_vec = new AeroStructProduct(this);  
  (*mat_vec)(u, w);
  delete mat_vec;

  double forward_prod = InnerProd(v, w);

  kona::MatrixVectorProduct<InnerProdVector>* 
      trans_mat_vec = new AeroStructTransposeProduct(this);  
  (*trans_mat_vec)(v, w);
  delete trans_mat_vec;  
  
  double reverse_prod = InnerProd(w, u);
  
  cout << "TestMDATranposedProduct: "
       << "error between forward and reverse products: "
       << forward_prod - reverse_prod << endl;

  p_ref_ = ref_save;
}

// ======================================================================

void AeroStructMDA::PrintDisplacements()
{
  for (int i=0; i<num_nodes_; i++) {
    cout << "Node " << i << " :: X-Displacement " << u_(3*(num_nodes_+i)) << endl;
    cout << "Node " << i << " :: Y-Displacement " << u_(3*(num_nodes_+i)+1) << endl;
    cout << "Node " << i << " :: rotation " << u_(3*(num_nodes_+i)+2) << endl;
  }
}

// ======================================================================
// OPTIMIZATION ROUTINES
// ======================================================================

void AeroStructMDA::Calc_dRdB_Product(InnerProdVector & in, InnerProdVector & out)
{
  InnerProdVector wrk(num_nodes_, 0.0);
  nozzle_->AreaForwardDerivative(in, wrk);      // (dA/dB)*in
  cfd_.JacobianAreaProduct(wrk, out);           // (dR/dA)*(dA/dB)*in
}

void AeroStructMDA::CalcTrans_dRdB_Product(InnerProdVector & in, InnerProdVector & out)
{
  InnerProdVector wrk(num_nodes_, 0.0);
  cfd_.JacobianTransposedAreaProduct(in, wrk);  // (dR/dA)^T *in
  nozzle_->AreaReverseDerivative(wrk, out);     // (dA/dB)^T *(dR/dA)^T *in
}

void AeroStructMDA::Calc_dSdB_Product(InnerProdVector & in, InnerProdVector & out)
{
  InnerProdVector wrk1(num_nodes_, 0.0);
  nozzle_->AreaForwardDerivative(in, wrk1);     // (dA/dB)*in
  InnerProdVector wrk2(num_nodes_, 0.0);
  csm_.Calc_dydA_Product(wrk1, wrk2);           // (dy/dA)*(dA/dB)*in
  csm_.CalcFD_dSdy_Product(wrk2, out);          // (dS/dy)*(dy/dA)*(dA/dB)*in
}

void AeroStructMDA::CalcTrans_dSdB_Product(InnerProdVector & in, InnerProdVector & out)
{
  InnerProdVector wrk1(num_nodes_, 0.0);
  for (int i = 0; i < num_nodes_; i++)
    wrk1(i) = in(3*i+1); // extract y-coordinate locations from the input vector
  InnerProdVector wrk2(num_design_, 0.0);
  csm_.CalcTransFD_dSdy_Product(wrk1, wrk2);      // (dS/dy)^T *in
  InnerProdVector wrk3(num_nodes_, 0.0);
  csm_.CalcTrans_dydA_Product(wrk2, wrk3);      // (dy/dA)^T *(dS/du)^T *in
  nozzle_->AreaReverseDerivative(wrk3, out);    // (dA/dB)^T *(du/dA)^T *(dS/du)^T *in
}

void AeroStructMDA::AeroStructDesignProduct(InnerProdVector & in, InnerProdVector & out)
{
  InnerProdVector v_cfd(3*num_nodes_, 0.0);
  InnerProdVector v_csm(3*num_nodes_, 0.0);
  Calc_dRdB_Product(in, v_cfd);
  Calc_dSdB_Product(in, v_csm);
  for (int i = 0; i < 3*num_nodes_; i++) {
    out(i) = v_cfd(i);
    out((3*num_nodes_)+i) = v_csm(i);
  }
}

void AeroStructMDA::AeroStructDesignTransProduct(InnerProdVector & in, InnerProdVector & out)
{
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // THIS NEEDS REVIEW
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  InnerProdVector u_cfd(3*num_nodes_, 0.0);
  InnerProdVector u_csm(3*num_nodes_, 0.0);
  for (int i = 0; i < 3*num_nodes_; i++) {
    u_cfd(i) = in(i);
    u_csm(i) = in((3*num_nodes_)+i);
  }
  InnerProdVector wrk1(num_design_, 0.0);
  CalcTrans_dRdB_Product(in, wrk1);
  CalcTrans_dSdB_Product(in, out);
  out += wrk1;
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
  wrk *= mda_->p_ref_;
  // NOTE: below, I assume u_cfd is not needed anymore so I can use it for work
  mda_->csm_.Calc_dSdp_Product(wrk, u_cfd);
  v_csm += u_cfd;

  // Finally, assemble v from its cfd and csm parts
  for (int i = 0; i < 3*nnp; i++) {
    v(i) = v_cfd(i);
    v(3*nnp+i) = v_csm(i);
  }

  // Scale product
  mda_->ScaleVector(v);
#if 0
  // TEMP: identity operator (relaxation)
  v = u;
#endif
}

// ======================================================================

void AeroStructTransposeProduct::operator()(const InnerProdVector & u, 
                                            InnerProdVector & v) 
{
  // decompose u into its cfd and csm parts, and create some work arrays
  int nnp = mda_->num_nodes_;
  InnerProdVector u_cfd(3*nnp, 0.0), v_cfd(3*nnp, 0.0),
      u_csm(3*nnp, 0.0), v_csm(3*nnp, 0.0),
      wrk(nnp, 0.0);

  // Scale input
  v = u;
  //mda_->ScaleVector(v);  
  for (int i = 0; i < 3*nnp; i++) {
    u_cfd(i) = v(i);
    u_csm(i) = v(3*nnp+i);
  }
  
  // Denote the Aerostructural Jacobian-vector product by
  //    | A^T  C^T | | u_cfd | = | v_cfd |
  //    | B^T  D   | | u_csm |   | v_csm |

  // Compute A^T*u_cfd
  mda_->cfd_.JacobianTransposedStateProduct(u_cfd, v_cfd);

  // Compute D^T*u_csm
  mda_->csm_.Calc_dSdu_Product(u_csm, v_csm);

  // Compute B^T*u_cfd = (dA/du)^T*(dR/dA)^T*u_cfd = (dA/du)^T*wrk
  mda_->cfd_.JacobianTransposedAreaProduct(u_cfd, wrk);
  // NOTE: below, I assume u_cfd is not needed anymore, so I can use it for work
  mda_->csm_.CalcTrans_dAdu_Product(wrk, u_cfd);
  v_csm += u_cfd;

  // Compute C^T*u_csm = (dp/dq)^T*(dS/dp)^T*u_csm = (dp/dq)^T*wrk
  mda_->csm_.CalcTrans_dSdp_Product(u_csm, wrk);
  // NOTE: below, I assume u_cfd is not needed anymore so I can use it for work
  mda_->cfd_.CalcDPressDQTransposedProduct(wrk, u_cfd);
  u_cfd *= mda_->p_ref_;
  v_cfd += u_cfd;
  
  // Finally, assemble v from its cfd and csm parts
  for (int i = 0; i < 3*nnp; i++) {
    v(i) = v_cfd(i);
    v(3*nnp+i) = v_csm(i);
  }
#if 0
  // TEMP: identity operator (relaxation)
  v = u;
#endif
}

// ======================================================================

void AeroStructPrecond::operator()(InnerProdVector & u, InnerProdVector & v)
{
  // decompose u into its cfd and csm parts, and create some work arrays
  int nnp = mda_->num_nodes_;
  InnerProdVector u_cfd(3*nnp, 0.0), u_csm(3*nnp, 0.0), wrk(nnp, 0.0),
      v_cfd(3*nnp, 0.0), v_csm(3*nnp, 0.0);
  for (int i = 0; i < 3*nnp; i++) {
    u_cfd(i) = u(i);
    u_csm(i) = u(3*nnp+i);
  }

#if 0
  // Compute v_cfd = M^{-1}(u_cfd - B*u_csm)
  mda_->csm_.Calc_dAdu_Product(v_csm, wrk);
  mda_->cfd_.JacobianAreaProduct(wrk, v_cfd);
  u_cfd.EqualsAXPlusBY(1.0, u_cfd, -1.0, v_cfd);
  
  // inherit the preconditioner calculated at every iteration for the CFD  
  mda_->cfd_.Precondition(u_cfd, v_cfd);

  // Compute v_csm = u_csm - C*v_cfd
  mda_->cfd_.CalcDPressDQProduct(v_cfd, wrk);
  wrk *= mda_->p_ref_;
  mda_->csm_.Calc_dSdp_Product(wrk, u_cfd);
  u_csm -= u_cfd;
  mda_->csm_.Precondition(u_csm, v_csm);
#else

  mda_->csm_.SolveFor(u_csm, 100, 1e-5);
  v_csm = mda_->csm_.get_u();
  mda_->cfd_.Precondition(u_cfd, v_cfd);

#endif
  // merge the preconditioners and pass it up
  for (int i = 0; i < 3*nnp; i++) {
    v(i) = v_cfd(i);
    v(3*nnp+i) = v_csm(i);
  }
#if 0
  // TEMP: identity preconditioner
  v = u;
#endif
}

// ======================================================================

void AeroStructTransposePrecond::operator()(InnerProdVector & u, InnerProdVector & v)
{
  // decompose u into its cfd and csm parts, and create some work arrays
  int nnp = mda_->num_nodes_;
  InnerProdVector u_cfd(3*nnp, 0.0), u_csm(3*nnp, 0.0), wrk(nnp, 0.0),
      v_cfd(3*nnp, 0.0), v_csm(3*nnp, 0.0);
  for (int i = 0; i < 3*nnp; i++) {
    u_cfd(i) = u(i);
    u_csm(i) = u(3*nnp+i);
  }

#if 0
  // Compute v_cfd = M^{-1}(u_cfd - B*u_csm)
  mda_->csm_.Calc_dAdu_Product(v_csm, wrk);
  mda_->cfd_.JacobianAreaProduct(wrk, v_cfd);
  u_cfd.EqualsAXPlusBY(1.0, u_cfd, -1.0, v_cfd);
  
  // inherit the preconditioner calculated at every iteration for the CFD  
  mda_->cfd_.Precondition(u_cfd, v_cfd);

  // Compute v_csm = u_csm - C*v_cfd
  mda_->cfd_.CalcDPressDQProduct(v_cfd, wrk);
  wrk *= mda_->p_ref_;
  mda_->csm_.Calc_dSdp_Product(wrk, u_cfd);
  u_csm -= u_cfd;
  mda_->csm_.Precondition(u_csm, v_csm);
#else

  mda_->csm_.SolveFor(u_csm, 100, 1e-5);
  v_csm = mda_->csm_.get_u();
  mda_->cfd_.PreconditionTransposed(u_cfd, v_cfd);
  
#endif
  // merge the preconditioners and pass it up
  for (int i = 0; i < 3*nnp; i++) {
    v(i) = v_cfd(i);
    v(3*nnp+i) = v_csm(i);
  }
#if 0
  // TEMP: identity preconditioner
  v = u;
#endif
}
