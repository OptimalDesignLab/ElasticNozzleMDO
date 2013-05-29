/**
 * \file idf.cpp
 * \brief driver for a simple test of IDF solved using RSNK
 * \version 1.0
 */

#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include <user_memory.hpp>
#include <kona.hpp>
#include <boost/math/constants/constants.hpp>

#include "../quasi_1d_euler/inner_prod_vector.hpp"
#include "../quasi_1d_euler/exact_solution.hpp"
#include "../quasi_1d_euler/nozzle.hpp"
#include "../quasi_1d_euler/quasi_1d_euler.hpp"
#include "../linear_elastic_csm/lecsm.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::ofstream;

const double pi = boost::math::constants::pi<double>();

// global variables...but only to this file
static int num_design_vec = -1; // number of design vectors
static int num_state_vec = -1;  // number of state vectors
static int num_dual_vec = -1;   // number of dual vectors
static vector<InnerProdVector> design;    // design vectors
static vector<InnerProdVector> state;     // state vectors
static vector<InnerProdVector> dual;      // dual vectors
 
// domain parameters
static const double length = 1.0;
static const double x_min = 0.0;
static const double x_max = x_min + length;

// design parameters
static int num_design; // total number of design vars (control points + coupling)
static int num_bspline; // number of b-spline control points
static const double area_left = 2.0;
static const double area_right = 1.75;

// discretization parameters
static const int nodes = 21;
static const int num_dis_var = 3*nodes; // number of states in a discipline
static const int num_var = 2*num_dis_var; // number of states total
static const int num_ceq = 2*nodes; // number of equality constraints

// CFD discretization parameters
static const int order = 3;
static const bool sbp_quad = true; // use SBP norm for quadrature

// CFD solution parameters
static const double kAreaStar = 0.8; // for the exact solution
static const bool subsonic = true; // for the exact solution
static const double kTempStag = 300.0;
static const double kPressStag = 100000;
static const double kRGas = 287.0;
static double rho_R, rho_u_R, e_R;
static double p_ref;
static const InnerProdVector press_stag(nodes, kPressStag);

// CSM solution parameters
double E = 100000000.0;   // Young's modulus
double w = 1.0;         // fixed width of nozzle
double t = 0.01;        // fixed beam element thickness
double h = 2;           // max height of the nozzle

static LECSM csm_solver(nodes);
static Quasi1DEuler cfd_solver(nodes, order);
static BsplineNozzle nozzle_shape;

double MeshCoord(const double & length, const int & num_nodes,
                 const int & i);

double TargetNozzleArea(const double & x);

double InitNozzleArea(const double & x);

void InitCFDSolver(const InnerProdVector & x_coord,
                   const InnerProdVector & press_targ);

void InitCSMSolver(const InnerProdVector & x_coord,
                   const InnerProdVector & y_coord);

void CalcYCoord(const InnerProdVector & area, InnerProdVector y_coord);

void CalcDAreaDYdisp(const InnerProdVector & u_csm, InnerProdVector & v_area);

void CalcDAreaDYdispTrans(const InnerProdVector & u_area,
                          InnerProdVector & v_csm);

// The following routines extract specific vectors from design[i]
void GetBsplinePts(const int & i, InnerProdVector & pts);
void GetCouplingPress(const int & i, InnerProdVector & press);
void GetCouplingArea(const int & i, InnerProdVector & area);

// The following routines set specific vectors in design[i]
void SetBsplinePts(const int & i, const InnerProdVector & pts);
void SetCouplingPress(const int & i, const InnerProdVector & press);
void SetCouplingArea(const int & i, const InnerProdVector & area);

// The following routines extract q and u from state[i]
void GetCFDState(const int & i, InnerProdVector & q);
void GetCSMState(const int & i, InnerProdVector & u);

// The following routines set q and u in state[i]
void SetCFDState(const int & i, const InnerProdVector & q);
void SetCSMState(const int & i, const InnerProdVector & u);

// The following routines extract parts of the constraints from dual[i]
void GetPressCnstr(const int & i, InnerProdVector & ceq_press);
void GetAreaCnstr(const int & i, InnerProdVector & ceq_area);

// The following routines set parts of the constraints in dual[i]
void SetPressCnstr(const int & i, const InnerProdVector & ceq_press);
void SetAreaCnstr(const int & i, const InnerProdVector & ceq_area);

int userFunc(int request, int leniwrk, int *iwrk, int lendwrk,
	     double *dwrk);

// ======================================================================

int main(int argc, char *argv[]) {

  if (argc != 2) {
    cerr << "Error in design: expect exactly one command line variable "
         << "(number of B-spline control points defining nozzle area)"
         << endl;
  } else {
    num_bspline = atoi(argv[1]);
    num_design = num_bspline + 2*nodes;
    cout << "Running design with " << num_design << " design vars." << endl;
  }

  // define the x-coordinates
  InnerProdVector x_coord(nodes, 0.0);
  for (int i = 0; i < nodes; i++) {
    x_coord(i) = MeshCoord(length, nodes, i);
    double A = area_left + (area_right - area_left)*(x_coord(i)/length);
    y_coord(i) = 0.5*(h - A/w);
  }
  
  // define the target nozzle shape and set area 
  // NOTE: the left and right nozzle areas are fixed and are not
  // included in the number of design variables
  InnerProdVector area_targ(nodes, 0.0);
  for (int i = 0; i < nodes; i++)
    area_targ(i) = TargetNozzleArea(x_coord(i));  
  nozzle_shape.SetAreaAtEnds(area_left, area_right);
  
  // find the target pressure and the number of preconditioner calls for the MDA
  InnerProdVector press_targ(nodes, 0.0);
  int solver_precond_calls = FindTargPress(press_targ);

  // set-up the cfd_solver
  InitCFDSolver(x_coord, press_targ);
  
  // set-up the csm_solver
  InitCSMSolver(x_coord);
  
  map<string,string> optns;
  //optns["inner.lambda_init"] = "0.8";
  KonaOptimize(userFunc, optns);
  int kona_precond_calls = cfd_solver.TotalPreconditionerCalls();
    
  cout << "Total cfd_solver precond calls: " << solver_precond_calls << endl;
  cout << "Total Kona precond calls:   " << kona_precond_calls << endl;
  cout << "Ratio of precond calls:     " 
       << ( static_cast<double>(kona_precond_calls)
            /static_cast<double>(solver_precond_calls) ) << endl;

  // output the optimized nozzle and flow
  //cfd_solver.WriteTecplot(rho_ref, a_ref);
  cfd_solver.WriteTecplot(1.0, 1.0, "flow_opt.dat");
  double error_L2, error_inf;
  cfd_solver.CalcMachError(kAreaStar, subsonic, error_L2,
                       error_inf);
  cout << "error L2 = " << error_L2 << endl;
  cout << "error_inf = " << error_inf << endl;
}

// ======================================================================

double MeshCoord(const double & length, const int & num_nodes,
                 const int & i) {
  double xi = static_cast<double>(i)/static_cast<double>(num_nodes-1);
  // uniform spacing
  return length*xi;
  // simple exponential mesh spacing
  //return (exp(4.0*xi)-1.0)/(exp(4.0)-1.0);
  // mesh ponts clustered near center
  //return (xi + (cos(pi*xi) - 1.0)/pi)/(1.0 -2.0/pi);
}

// ======================================================================

double TargetNozzleArea(const double & x) {
  // cubic polynomial nozzle
  const double area_mid = 1.5;
  double a = area_left;
  double b = 4.0*area_mid - 5.0*area_left + area_right;
  double c = -4.0*(area_right -2.0*area_left + area_mid);
  double d = 4.0*(area_right - area_left);
  return a + x*(b + x*(c + x*d));
}

// ======================================================================

double InitNozzleArea(const double & x) {
  // linear nozzle
  return area_left + (area_right - area_left)*x;
}

// ======================================================================

void FindTargPress(InnerProdVector & targ_press) {
  // WARNING: we are hard coding the parameters into InitializeTestProb()!!!
  AeroStructMDA asmda(nodes, order);
  asmda.InitializeTestProb();
  int precond_calls = asmda.NewtonKrylov(30, 1.e-8);
  
}

// ======================================================================

void InitCFDSolver(const InnerProdVector & x_coord,
                   const InnerProdVector & press_targ) {
  // define reference and boundary conditions
  double rho, rho_u, e;
  CalcFlowExact(kGamma, kRGas, kAreaStar, area_left, true, 
                kTempStag, kPressStag, rho, rho_u, e);
  double rho_ref = rho;
  double press = (kGamma - 1.0)*(e - 0.5*rho_u*rho_u/rho);
  double p_ref = press;
  double a_ref = sqrt(kGamma*press/rho_ref);
  double rho_L = 1.0;
  double rho_u_L = rho_u/(a_ref*rho_ref);
  double e_L = e/(rho_ref*a_ref*a_ref);
  CalcFlowExact(kGamma, kRGas, kAreaStar, area_right, true,
                kTempStag, kPressStag, rho, rho_u, e);
  rho_R = rho/rho_ref;
  rho_u_R = rho_u/(a_ref*rho_ref);
  e_R =  e/(rho_ref*a_ref*a_ref); 

  // set boundary and initial conditions
  cfd_solver.set_bc_left(rho_L, rho_u_L, e_L);
  cfd_solver.InitialCondition(rho_R, rho_u_R, e_R);
  cfd_solver.set_bc_right(rho_R, rho_u_R, e_R);

  // define x-coordinates
  cfd_solver.set_x_coord(x_coord);

  // define area?
  
  // define any discretization and cfd_solver paramters
  cfd_solver.set_diss_coeff(0.04);

  // define target pressure
  cfd_solver.set_press_targ(press_targ);
}

// ======================================================================

void InitCSMSolver(const InnerProdVector & x_coord,
                   const InnerProdVector & y_coord) {
  // set material properties
  csm_solver.set_material(E, t, w, h);
  // create the CSM mesh
  csm_solver.GenerateMesh(x_coord, y_coord);
  
  // set the nodal degrees of freedom
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
  csm_.SetBoundaryConds(BCtype, BCval);  
}

// ======================================================================

void CalcYCoord(const InnerProdVector & area, InnerProdVector y_coord) {
  assert(area.size() == nodes);
  assert(y_coord.size() == nodes);
  for (int j = 0; j < nodes; j++)
    y(j) = 0.5*(h - area(j)/w);  
}

// ======================================================================

void CalcDAreaDYdisp(const InnerProdVector & u_csm, InnerProdVector & v_area) {
  assert(u_csm.size() == num_dis_var);
  assert(v_area.size() == nodes);
  for (int j = 0; j < nodes; j++)
    v_area(j) = -2.0*w*u_csm(3*j+1);
}

// ======================================================================

void CalcDAreaDYdispTrans(const InnerProdVector & u_area,
                          InnerProdVector & v_csm) {
  assert(v_csm.size() == num_dis_var);
  assert(u_area.size() == nodes);
  for (int j = 0; j < nodes; j++)
    v_csm(3*j+1) = -2.0*w*u_area(j);
}

// ======================================================================

void GetBsplinePts(const int & i, InnerProdVector & pts) {
  assert((i >= 0) && (i < num_design_vec));
  assert(pts.size() == num_bspline);
  for (int j = 0; j < num_bspline; j++)
    pts(j) = design[i](j);  
}

// ======================================================================

void GetCouplingPress(const int & i, InnerProdVector & press) {
  assert((i >= 0) && (i < num_design_vec));
  assert(press.size() == nodes);
  for (int j = 0; j < nodes; j++)
    press(j) = design[i](num_bspline+j);
}

// ======================================================================

void GetCouplingArea(const int & i, InnerProdVector & area) {
  assert((i >= 0) && (i < num_design_vec));
  assert(area.size() == nodes);
  for (int j = 0; j < nodes; j++)
    area(j) = design[i](num_bspline + nodes + j);
}

// ======================================================================

void SetBsplinePts(const int & i, const InnerProdVector & pts) {
  assert((i >= 0) && (i < num_design_vec));
  assert(pts.size() == num_bspline);
  for (int j = 0; j < num_bspline; j++) {
    design[i](j) = pts(j);
  }
}

// ======================================================================

void SetCouplingPress(const int & i, const InnerProdVector & press) {
  assert((i >= 0) && (i < num_design_vec));
  assert(press.size() == nodes);
  for (int j = 0; j < nodes; j++)
    design[i](num_bspline+j) = press(j);
}

// ======================================================================

void SetCouplingArea(const int & i, const InnerProdVector & area) {
  assert((i >= 0) && (i < num_design_vec));
  assert(area.size() == nodes);
  for (int j = 0; j < nodes; j++)
    design[i](num_bspline + nodes + j) = area(j);
}

// ======================================================================

void GetCFDState(const int & i, InnerProdVector & q) {
  assert((i >= 0) && (i < num_state_vec));
  assert(q.size() == num_dis_var);
  for (int j = 0; j < num_dis_var; j++)
    q(j) = state[i](j);
}

// ======================================================================

void GetCSMState(const int & i, InnerProdVector & u) {
  assert((i >= 0) && (i < num_state_vec));
  assert(u.size() == num_dis_var);
  for (int j = 0; j < num_dis_var; j++)
    u(j) = state[i](num_dis_var+j);
}

// ======================================================================

void SetCFDState(const int & i, const InnerProdVector & q) {
  assert((i >= 0) && (i < num_state_vec));
  assert(q.size() == num_dis_var);
  for (int j = 0; j < num_dis_var; j++)
    state[i](j) = q(j);
}

// ======================================================================

void SetCSMState(const int & i, const InnerProdVector & u) {
  assert((i >= 0) && (i < num_state_vec));
  assert(u.size() == num_dis_var);
  for (int j = 0; j < num_dis_var; j++)
    state[i](num_dis_var+j) = u(j);
}

// ======================================================================

void GetPressCnstr(const int & i, InnerProdVector & ceq_press) {
  assert((i >= 0) && (i < num_dual_vec));
  assert(ceq_press.size() == nodes);
  for (int j = 0; j < nodes; j++)
    ceq_press(j) = dual[i](j);
}

// ======================================================================

void GetAreaCnstr(const int & i, InnerProdVector & ceq_area) {
  assert((i >= 0) && (i < num_dual_vec));
  assert(ceq_area.size() == nodes);
  for (int j = 0; j < nodes; j++)
    ceq_area(j) = dual[i](nodes + j);
}

// ======================================================================

void SetPressCnstr(const int & i, const InnerProdVector & ceq_press) {
  assert((i >= 0) && (i < num_dual_vec));
  assert(ceq_press.size() == nodes);
  for (int j = 0; j < nodes; j++)
    dual[i](j) = ceq_press(j);
}

// ======================================================================

void SetAreaCnstr(const int & i, const InnerProdVector & ceq_area) {
  assert((i >= 0) && (i < num_dual_vec));
  assert(ceq_area.size() == nodes);
  for (int j = 0; j < nodes; j++)
     dual[i](nodes + j) = ceq_area(j);
}

// ======================================================================

int userFunc(int request, int leniwrk, int *iwrk, int lendwrk,
	     double *dwrk) {
  static int opt_iter = 0;
  switch (request) {
    case kona::allocmem: {// allocate iwrk[0] design vectors and
      // iwrk[1] state vectors
      if (num_design_vec >= 0) { // free design memory first
        if (design.size() == 0) {
          cerr << "userFunc: "
               << "design array is empty but num_design_vec > 0" << endl;
          throw(-1);
        }
        design.clear();
      }
      if (num_state_vec >= 0) { // free state memory first
        if (state.size() == 0) {
          cerr << "userFunc: "
               << "state array is empty but num_state_vec > 0" << endl;
          throw(-1);
        }
        state.clear();
      }
      if (num_dual_vec >= 0) { // free dual memory first
        if (dual.size() == 0) {
          cerr << "userFunc: "
               << "dual array is empty but num_dual_vec > 0" << endl;
          throw(-1);
        }
        dual.clear();
      }      
      num_design_vec = iwrk[0];
      num_state_vec = iwrk[1];
      num_dual_vec = iwrk[2];
      assert(num_design_vec >= 0);
      assert(num_state_vec >= 0);
      assert(num_dual_vec >= 0);
      design.resize(num_design_vec);
      for (int i = 0; i < num_design_vec; i++)
        design[i].resize(num_design);
      state.resize(num_state_vec);
      for (int i = 0; i < num_state_vec; i++)
        state[i].resize(num_var);
      dual.resize(num_dual_vec);
      for (int i = 0; i < num_dual_vec; i++)
        dual[i].resize(num_ceq);      
      break;
    }
    case kona::axpby_d: {// using design array set
      // iwrk[0] = dwrk[0]*iwrk[1] + dwrk[1]*iwrk[2]
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      assert((i >= 0) && (i < num_design_vec));
      assert(j < num_design_vec);
      assert(k < num_design_vec);
      
      double scalj = dwrk[0];
      double scalk = dwrk[1];
      if (j == -1) {
        if (k == -1) { // if both indices = -1, then all elements = scalj
          design[i] = scalj;
          
        } else { // if just j = -1 ...
          if (scalk == 1.0) {
            // direct copy of vector k with no scaling
            design[i] = design[k];
            
          } else {
            // scaled copy of vector k
            design[i] = design[k];
            design[i] *= scalk;
          }
        }
      } else if (k == -1) { // if just k = -1 ...
        if (scalj == 1.0) {
          // direct copy of vector j with no scaling
          design[i] = design[j];
          
        } else {
          // scaled copy of vector j
          design[i] = design[j];
          design[i] *= scalj;
        }
      } else { // otherwise, full axpby
        design[i].EqualsAXPlusBY(scalj, design[j], scalk, design[k]);
      }
      break;
    }
    case kona::axpby_s: {// using state array set
      // iwrk[0] = dwrk[0]*iwrk[1] + dwrk[1]*iwrk[2]
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      assert((i >= 0) && (i < num_state_vec));
      assert(j < num_state_vec);
      assert(k < num_state_vec);

      double scalj = dwrk[0];
      double scalk = dwrk[1];
      if (j == -1) {
        if (k == -1) { // if both indices = -1, then all elements = scalj
          state[i] = scalj;

        } else { // if just j = -1 ...
          if (scalk == 1.0) {
            // direct copy of vector k with no scaling
            state[i] = state[k];

          } else {
            // scaled copy of vector k
            state[i] = state[k];
            state[i] *= scalk;
          }
        }
      } else if (k == -1) { // if just k = -1 ...
        if (scalj == 1.0) {
          // direct copy of vector j with no scaling
          state[i] = state[j];

        } else {
          // scaled copy of vector j
          state[i] = state[j];
          state[i] *= scalj;
        }
      } else { // otherwise, full axpby
        state[i].EqualsAXPlusBY(scalj, state[j], scalk, state[k]);
      }
      break;
    }
    case kona::axpby_c: {// using dual array set
      // iwrk[0] = dwrk[0]*iwrk[1] + dwrk[1]*iwrk[2]
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      assert((i >= 0) && (i < num_dual_vec));
      assert(j < num_dual_vec);
      assert(k < num_dual_vec);

      double scalj = dwrk[0];
      double scalk = dwrk[1];
      if (j == -1) {
        if (k == -1) { // if both indices = -1, then all elements = scalj
          dual[i] = scalj;

        } else { // if just j = -1 ...
          if (scalk == 1.0) {
            // direct copy of vector k with no scaling
            dual[i] = dual[k];

          } else {
            // scaled copy of vector k
            dual[i] = dual[k];
            dual[i] *= scalk;
          }
        }
      } else if (k == -1) { // if just k = -1 ...
        if (scalj == 1.0) {
          // direct copy of vector j with no scaling
          dual[i] = dual[j];

        } else {
          // scaled copy of vector j
          dual[i] = dual[j];
          dual[i] *= scalj;
        }
      } else { // otherwise, full axpby
        dual[i].EqualsAXPlusBY(scalj, dual[j], scalk, dual[k]);
      }
      break;
    }      
    case kona::innerprod_d: {// using design array set
      // dwrk[0] = (iwrk[0])^{T} * iwrk[1]
      int i = iwrk[0];
      int j = iwrk[1];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_design_vec));
      dwrk[0] = InnerProd(design[i], design[j]);
      break;
    } 
    case kona::innerprod_s: {// using state array set
      // dwrk[0] = (iwrk[0])^{T} * iwrk[1]
      int i = iwrk[0];
      int j = iwrk[1];
      assert((i >= 0) && (i < num_state_vec));
      assert((j >= 0) && (j < num_state_vec));
      dwrk[0] = InnerProd(state[i], state[j]);
      break;
    }
    case kona::innerprod_c: {// using dual array set
      // dwrk[0] = (iwrk[0])^{T} * iwrk[1]
      int i = iwrk[0];
      int j = iwrk[1];
      assert((i >= 0) && (i < num_dual_vec));
      assert((j >= 0) && (j < num_dual_vec));
      dwrk[0] = InnerProd(dual[i], dual[j]);
      break;
    }      
    case kona::eval_obj: {// evaluate the objective
      int i = iwrk[0];
      int j = iwrk[1];      
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= -1) && (j < num_state_vec));
      InnerProdVector pts(num_bspline, 0.0);
      GetBsplinePts(i, pts);
      nozzle_shape.SetCoeff(pts);
      InnerProdVector area(nodes, 0.0);
      GetCouplingArea(i, area);
      cfd_solver.set_area(area);
      if (j == -1) {
        // need to solve for the state first
	cfd_solver.InitialCondition(rho_R, rho_u_R, e_R);
        iwrk[0] = cfd_solver.NewtonKrylov(100, 1.e-6);
      } else {
        iwrk[0] = 0; // no precondition calls
        InnerProdVector q(num_dis_var, 0.0);
        GetCFDState(j, q);
        cfd_solver.set_q(q);
      }
      dwrk[0] = cfd_solver.CalcInverseDesign();
      break;
    }
    case kona::eval_pde: {// evaluate PDE at (design,state) =
                         // (iwrk[0],iwrk[1])
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      // Evaluate the CFD residual
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      cfd_solver.set_area(area);
      cfd_solver.set_q(q);
      cfd_solver.CalcResidual();
      SetCFDState(k, cfd_solver.get_res());
      // Evaluate the CSM residual
      InnerProdVector press(nodes, 0.0), pts(num_bspline, 0.0),
          y_coords(nodes, 0.0), u(num_dis_var, 0.0);
      GetBsplinePts(i, pts);
      GetCouplingPress(i, press);
      GetCSMState(j, u);
      nozzle_shape.SetCoeff(pts);
      area = nozzle_shape.Area(cfd_solver.get_x_coord());
      CalcYCoords(area, y_coords);
      csm_solver.GenerateMesh(cfd_solver.get_x_coord(), y_coords);      
      press *= p_ref_;
      press -= press_stag;
      csm_solver.set_press(press);
      csm_solver.set_u(u);
      csm_solver.CalcResidual();
      SetCSMState(k, csm_solver.get_res());
      break;
    }
    case kona::eval_ceq: {// evaluate eq. constraint at (design,state) = (i,j)
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_dual_vec));
      // Evaluate the (press - press_coupling) constraint
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
          ceq_cfd(nodes, 0.0), press(nodes, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      cfd_solver.set_area(area);
      //cfd_solver.set_q(q);
      cfd_solver.CalcAuxiliaryVariables(q);
      ceq_cfd = cfd_solver.get_press();
      GetCouplingPress(i, press);
      ceq_cfd -= press;
      SetPressCnstr(k, ceq_cfd);
      // Evaluate the (area - area_coupling) constraint
      InnerProdVector pts(num_bspline, 0.0), u(num_dis_var, 0.0),
          y_coords(nodes, 0.0), ceq_csm(nodes, 0.0);
      GetBsplinePts(i, pts);
      GetCSMState(j, u);
      nozzle_shape.SetCoeff(pts);
      area = nozzle_shape.Area(cfd_solver.get_x_coord());
      CalcYCoords(area, y_coords);
      csm_solver.GenerateMesh(cfd_solver.get_x_coord(), y_coords); 
      csm_solver.set_u(u);
      csm_solver.CalcCoordsAndArea();
      ceq_csm = csm_solver.get_area();
      GetCouplingArea(i, area);
      ceq_csm -= area;
      SetAreaCnstr(k, ceq_csm);
      break;
    } 
    case kona::jacvec_d: {// apply design component of the Jacobian-vec
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_design_vec));
      assert((m >= 0) && (m < num_state_vec));
      // Evaluate the CFD part of the Jacobian-vec product
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
          u_area(nodes, 0.0), v_cfd(num_dis_var, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      GetCouplingArea(k, u_area); // u_area is the component being multiplied
      cfd_solver.set_area(area);
      cfd_solver.set_q(q);
      cfd_solver.JacobianAreaProduct(u_area, v_cfd);
      SetCFDState(m, v_cfd);
      // Evaluate the CSM part of the Jacobian-vec product
      InnerProdVector pts(num_bspline, 0.0), press(nodes, 0.0),
          y_coords(nodes, 0.0), v_csm(num_dis_var, 0.0);
      GetBsplinePts(i, pts);
      GetCouplingPress(i, press);
      GetCSMState(j, u);
      nozzle_shape.SetCoeff(pts);
      area = nozzle_shape.Area(cfd_solver.get_x_coord());
      CalcYCoords(area, y_coords);
      csm_solver.GenerateMesh(cfd_solver.get_x_coord(), y_coords);      
      press *= p_ref_;
      press -= press_stag;
      csm_solver.set_press(press);
      csm_solver.set_u(u);

      // first part: dS/dp * u_press
      GetCouplingPress(k, press);
      press *= p_ref;
      csm_solver.Calc_dSdp_Product(press, v_csm);

      // second part: dS/d(pts) * u_pts
      GetBsplinePts(k, pts);
      u_area = nozzle_shape.AreaForwardDerivative(cfd_solver.get_x_coord(),
                                                  pts);
      u_area /= -(2.0*w); // multiply dy/dA *(dA/db * u_pts)
      CalcFD_dSdy_Product(u_area, v_cfd); // multiply dS/dy *(dy/db * u_pts)
      v_csm += v_cfd;
      SetCSMState(m, v_csm);
      break;
    }
    case kona::jacvec_s: {// apply state component of the Jacobian-vector
      // product to vector iwrk[2]
      // recall; iwrk[0], iwrk[1] denote where Jacobian is evaluated
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      assert((m >= 0) && (m < num_state_vec));
      // Evaluate the CFD part of the Jacobian-vec product
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
          u_cfd(num_dis_var, 0.0), v_cfd(num_dis_var, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      GetCFDState(k, u_cfd); // u_cfd is the component being multiplied
      cfd_solver.set_area(area);
      cfd_solver.set_q(q);
      cfd_solver.JacobianStateProduct(u_cfd, v_cfd);
      SetCFDState(m, v_cfd);
      // Evaluate the CSM part of the Jacobian-vec product
      InnerProdVector pts(num_bspline, 0.0), press(nodes, 0.0),
          y_coords(nodes, 0.0), u_csm(num_dis_var, 0.0), v_csm(num_dis_var, 0.0);
      // Note: stiffness matrix does not depend on press
      GetBsplinePts(i, pts);
      nozzle_shape.SetCoeff(pts);
      area = nozzle_shape.Area(cfd_solver.get_x_coord());
      CalcYCoords(area, y_coords);
      csm_solver.GenerateMesh(cfd_solver.get_x_coord(), y_coords);      
      GetCSMState(k, u_csm);
      csm_solver.Calc_dSdu_Product(u_csm, v_csm);
      SetCSMState(m, v_csm);
      break;
    }
    case kona::tjacvec_d: {// apply design component of Jacobian to adj
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      assert((m >= 0) && (m < num_design_vec));
      // Evaluate the CFD part of the transposed-Jacobian-vec product
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
          u_cfd(num_dis_var, 0.0), v_area(nodes, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      GetCFDState(k, u_cfd); // u_cfd is the component being multiplied
      cfd_solver.JacobianTransposedAreaProduct(u_cfd, v_area);
      SetCouplingArea(m, v_area);
      // Evaluate the CSM part of the transposed-Jacobian-vec product
      InnerProdVector pts(num_bspline, 0.0), press(nodes, 0.0),
          y_coords(nodes, 0.0), u(num_dis_var, 0.0), v_press(nodes, 0.0),
          v_pts(num_bspline, 0.0), v_area(nodes, 0.0);
      GetBsplinePts(i, pts);
      GetCouplingPress(i, press);
      GetCSMState(j, u);
      nozzle_shape.SetCoeff(pts);
      area = nozzle_shape.Area(cfd_solver.get_x_coord());
      CalcYCoords(area, y_coords);
      csm_solver.GenerateMesh(cfd_solver.get_x_coord(), y_coords);      
      press *= p_ref_;
      press -= press_stag;
      csm_solver.set_press(press);
      csm_solver.set_u(u);

      // first part: (dS/dp)^T * u_csm
      GetCSMState(k, u);
      csm_solver.CalcTrans_dSdp_Product(u, v_press);
      v_press *= p_ref;
      SetCouplingPress(m, v_press);

      // second part: (dS/d(pts))^T * u_csm
      CalcTransFD_dSdy_Product(u, v_area); // (dS/dy)^T * u_csm
      v_area /= -(2.0*w); // (dy/dA)^T * (dS/dy)^T * u_csm
      v_pts = nozzle_shape.AreaReverseDerivative(
          cfd_solver.get_x_coord(), v_area); // (dA/db)^T * (dy/dA)^T ...
      SetBsplinePts(m, v_pts);
      break;
    }
    case kona::tjacvec_s: {// apply state component of Jacobian to adj
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      assert((m >= 0) && (m < num_state_vec));
      // Evaluate the CFD part of the tranposed-Jacobian-vec product
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
          u_cfd(num_dis_var, 0.0), v_cfd(num_dis_var, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      GetCFDState(k, u_cfd); // u_cfd is the component being multiplied
      cfd_solver.set_area(area);
      cfd_solver.set_q(q);
      cfd_solver.JacobianTransposedStateProduct(u_cfd, v_cfd);
      SetCFDState(m, v_cfd);
      // Evaluate the CSM part of the tranposed-Jacobian-vec product
      InnerProdVector pts(num_bspline, 0.0), y_coords(nodes, 0.0),
          u_csm(num_dis_var, 0.0), v_csm(num_dis_var, 0.0);
      // Note: stiffness matrix does not depend on press
      GetBsplinePts(i, pts);
      nozzle_shape.SetCoeff(pts);
      area = nozzle_shape.Area(cfd_solver.get_x_coord());
      CalcYCoords(area, y_coords);
      csm_solver.GenerateMesh(cfd_solver.get_x_coord(), y_coords);      
      GetCSMState(k, u_csm);
      csm_solver.Calc_dSdu_Product(u_csm, v_csm);
      SetCSMState(m, v_csm);            
      break;
    }
    case kona::eval_precond: {// build the preconditioner if necessary
      int i = iwrk[0];
      int j = iwrk[1];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      // Evaluate the CFD preconditioner
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      cfd_solver.set_area(area);
      cfd_solver.set_q(q);
      cfd_solver.BuildAndFactorPreconditioner();
      // Evaluate the CSM preconditioner?
      
      break;
    } 
    case kona::precond_s: {// apply primal preconditioner to iwrk[2]
      // recall; iwrk[0], iwrk[1] denote where preconditioner is
      // evaluated (CFD preconditioner has already been evaluated)
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      assert((m >= 0) && (m < num_state_vec));
      // Apply the CFD preconditioner
      InnerProdVector u_cfd(num_dis_var, 0.0), v_cfd(num_dis_var, 0.0);
      GetCFDState(k, u_cfd);
      cfd_solver.Precondition(u_cfd, v_cfd);
      SetCFDState(m, v_cfd);
      // Apply the CSM preconditioner
      InnerProdVector pts(num_bspline, 0.0), y_coords(nodes, 0.0),
          u_csm(num_dis_var, 0.0);
      GetBsplinePts(i, pts);
      nozzle_shape.SetCoeff(pts);
      area = nozzle_shape.Area(cfd_solver.get_x_coord());
      CalcYCoords(area, y_coords);
      csm_solver.GenerateMesh(cfd_solver.get_x_coord(), y_coords);
      GetCSMState(k, u_csm);
      csm_solver.SolveFor(u_csm, 100, 1e-5);
      SetCSMState(m, csm_solver.get_u());
      iwrk[0] = 1; // one preconditioner application
      break;
    }
    case kona::tprecond_s: {// apply adjoint preconditioner to iwrk[2]
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      assert((m >= 0) && (m < num_state_vec));
      // Apply the transposed CFD preconditioner
      InnerProdVector u_cfd(num_dis_var, 0.0), v_cfd(num_dis_var, 0.0);
      GetCFDState(k, u_cfd);
      cfd_solver.PreconditionTransposed(u_cfd, v_cfd);
      SetCFDState(m, v_cfd);
      // Apply the transposed CSM preconditioner
      InnerProdVector pts(num_bspline, 0.0), y_coords(nodes, 0.0),
          u_csm(num_dis_var, 0.0);
      GetBsplinePts(i, pts);
      nozzle_shape.SetCoeff(pts);
      area = nozzle_shape.Area(cfd_solver.get_x_coord());
      CalcYCoords(area, y_coords);
      csm_solver.GenerateMesh(cfd_solver.get_x_coord(), y_coords);
      GetCSMState(k, u_csm);
      csm_solver.SolveFor(u_csm, 100, 1e-5);
      SetCSMState(m, csm_solver.get_u());
      iwrk[0] = 1; // one preconditioner application
      break;
    }
    case kona::ceqjac_d: {// design component of equality constraint Jacobian
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_design_vec));
      assert((m >= 0) && (m < num_dual_vec));
      // component corresponding to (press - press_coupling)
      InnerProdVector u_press(nodes, 0.0);
      GetCouplingPress(k, u_press);
      u_press *= -1.0;
      SetPressCnstr(m, u_press);
      // component corresponding to (area - area_coupling)
      InnerProdVector u_area(nodes, 0.0), u_pts(num_bspline, 0.0);
      GetCouplingArea(k, u_area);
      u_area *= -1.0;
      GetBsplinePts(k, u_pts);     
      u_area += nozzle_shape.AreaForwardDerivative(cfd_solver.get_x_coord(),
                                                   u_pts);
      SetAreaCnstr(m, u_area);      
      break;
    }
    case kona::ceqjac_s: {// state component of equality constraint Jacobian
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      assert((m >= 0) && (m < num_dual_vec));
      // component corresponding to (press - press_coupling)
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
          u_cfd(num_dis_var, 0.0), v_press(nodes, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      cfd_solver.set_area(area);
      cfd_solver.set_q(q);
      GetCFDState(k, u_cfd);
      cfd_solver.CalcDPressDQProduct(u_cfd, v_press);
      SetPressCnstr(m, v_press);
      // component corresponding to (area - area_coupling)
      InnerProdVector u_csm(num_dis_var, 0.0), v_area(nodes, 0.0);
      GetCSMState(k, u_csm);
      CalcDAreaDYdisp(u_csm, v_area);
      SetAreaCnstr(m, v_area);      
      break;
    }
    case kona::tceqjac_d: {// apply design component of constraint Jac to dual
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_dual_vec));
      assert((m >= 0) && (m < num_design_vec));
      // component corresponding to (press - press_coupling)
      InnerProdVector u_press(nodes, 0.0);
      GetPressCnstr(k, u_press);
      u_press *= -1.0;
      SetCouplingPress(m, u_press);
      // component corresponding to (area - area_coupling)      
      InnerProdVector u_area(nodes, 0.0), v_pts(num_bspline, 0.0);
      GetAreaCnstr(k, u_area);
      u_area *= -1.0;
      SetCouplingArea(m, u_area);
      u_area *= -1.0;
      v_pts = nozzle_shape.AreaReverseDerivative(
          cfd_solver.get_x_coord(), u_area);
      SetBsplinePts(m, v_pts);
      break;
    }
    case kona::tceqjac_s: {// apply state component of constraint Jac to dual
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_dual_vec));
      assert((m >= 0) && (m < num_state_vec));
      // component corresponding to (press - press_coupling)
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
          u_press(nodes, 0.0), v_cfd(num_dis_var, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      cfd_solver.set_area(area);
      cfd_solver.set_q(q);
      GetPressCnstr(k, u_press);
      cfd_solver.CalcDPressDQTransposedProduct(u_press, v_cfd);
      SetCFDState(m, v_cfd);
      // component corresponding to (area - area_coupling)
      InnerProdVector u_area(nodes, 0.0), v_csm(num_dis_var, 0.0);
      GetAreaCnstr(k, u_area);
      CalcDAreaDydispTrans(u_area, v_csm);
      SetCSMState(m, v_csm);
      break;        
    }       
    case kona::grad_d: {// design component of objective gradient
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_design_vec));
      design[k] = 0.0;
      break;
    }
    case kona::grad_s: {// state component of objective gradient
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      // CFD contribution to objective gradient
      InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
          g_cfd(num_dis_var, 0.0);
      GetCouplingArea(i, area);
      GetCFDState(j, q);
      cfd_solver.set_area(area);
      cfd_solver.set_q(q);
      cfd_solver.CalcInverseDesigndJdQ(g_cfd);
      SetCFDState(k, g_cfd);
      // CSM contribution to objective gradient
      g_cfd = 0.0;
      SetCSMState(k, g_cfd);
      break;
    }
    case kona::initdesign: {// initialize the design variables
      int i = iwrk[0];
      assert((i >= 0) && (i < num_design_vec));      
      // in case the nozzle has not been initiated
      InnerProdVector pts(num_bspline, 0.0), press(nodes, 0.0),
          area(nodes, 0.0);
      pts = 1.0;

      cout << "beginning initdesign..." << endl;
      
      SetBsplinePts(i, pts);
      cout << "after SetBsplinePts(i, pts)" << endl;
      nozzle_shape.SetCoeff(pts);

      // fit a b-spline nozzle to a given shape
      InnerProdVector x_coord(nodes, 0.0);
      for (int j = 0; j < nodes; j++) {
        x_coord(j) = MeshCoord(length, nodes, j);
        area(j) = InitNozzleArea(x_coord(j)/length);
      }
      nozzle_shape.FitNozzle(x_coord, area);
      nozzle_shape.GetCoeff(pts);

      cout << "after initializing nozzle_shape..." << endl;
      SetBsplinePts(i, pts);
      SetCouplingArea(i, area);

      // Solve for pressure based on initial area (is there a better way?)
      cfd_solver.InitialCondition(rho_R, rho_u_R, e_R);
      iwrk[0] = cfd_solver.NewtonKrylov(100, 1.e-6);
      press = cfd_solver.get_press();
      SetCouplingPress(i, press);
      
      cout << "kona::initdesign design coeff = ";
      for (int n = 0; n < num_bspline; n++)
        cout << pts(n) << " ";
      cout << endl;
      
      break;
    }
    case kona::solve: { // solve the primal equations
      int i = iwrk[0];
      int j = iwrk[1];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      // Solve the CFD
      InnerProdVector area(nodes, 0.0);
      GetCouplingArea(i, area);
      cfd_solver.InitialCondition(rho_R, rho_u_R, e_R);
      iwrk[0] = cfd_solver.NewtonKrylov(20, 1.e-10);
      SetCFDState(j, cfd_solver.get_q());
      // Solve the CSM
      InnerProdVector press(nodes, 0.0), pts(num_bspline, 0.0);
      GetBsplinePts(i, pts);
      GetCouplingPress(i, press);
      nozzle_shape.SetCoeff(pts);
      area = nozzle_shape.Area(cfd_solver.get_x_coord());
      CalcYCoords(area, y_coords);
      csm_solver.GenerateMesh(cfd_solver.get_x_coord(), y_coords);
      press *= p_ref_;
      press -= press_stag;
      csm_solver.set_press(press);
      csm_solver.Solve();
      SetCSMState(j, csm_solver.get_u());
      // iwrk[0] += ?? how to incorporate?
      break;
    }
    case kona::adjsolve: {// solve the adjoint equations
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      // This should not be needed for IDF (the adjoint solve is done inside
      // kona)
      cerr << "Error in userFunc(): unexpected case kona::adjsolve!!!!" << endl;
      throw(-1);      
      break;
    }
    case kona::info: {// supplies information to user
      // current design is in iwrk[0]
      // current pde solution is in iwrk[1]
      // current adjoint solution is in iwrk[2]
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int iter = iwrk[3];
      InnerProdVector pts(num_bspline, 0.0);
      GetBsplinePts(i, pts);
      nozzle_shape.SetCoeff(pts);

#if 0
      // uncomment to list B-spline coefficients
      cout << "before kona::info set_area: design coeff = ";
      for (int n = 0; n < num_bspline; n++)
        cout << design[i](n) << " ";
      cout << endl;
#endif
      
#if 0
      cout << "total preconditioner calls (cfd_solver says) = " 
           << cfd_solver.TotalPreconditionerCalls() << endl;
#endif

#if 0
      string filename("flow_at_opt_iter");
      std::stringstream ss;
      ss << opt_iter;
      filename.append(ss.str());
      filename.append(".dat");
      //cfd_solver.WriteTecplot(rho_ref, a_ref, filename);
      cfd_solver.WriteTecplot(1.0, 1.0, filename);
      opt_iter++;
#endif
      break;
    }
    default: {
      cerr << "userFunc: "
           << "unrecognized request value: request = "
           << request << endl;
      throw(-1);
      break;
    }
  }
  return 0;
}
