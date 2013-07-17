/**
 * \file optimizer.cpp
 * \brief main binary for aero-structural MDA optimization
 * \author Jason Hicken <jason.hicken@gmail.com>, Alp Dener <alp.dener@gmail.com>
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

#include "../aerostruct.hpp"
#include "../constants.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::ofstream;

const double pi = boost::math::constants::pi<double>();

// optimizer-specific global variables
static int num_design_vec = -1; // number of design vectors
static int num_state_vec = -1;  // number of state vectors
static vector<InnerProdVector> design;    // design vectors
static vector<InnerProdVector> state;     // state vectors

// discretization parameters
static const int num_var = 6*nodes;
static int num_design;

static BsplineNozzle nozzle_shape;
static AeroStructMDA solver(nodes, order, nozzle_shape);

// forward declerations
int userFunc(int request, int leniwrk, int *iwrk, int lendwrk,
	     double *dwrk);
double InitNozzleArea(const double & x);
double TargetNozzleArea(const double & x);
double MeshCoord(const double & length, const int & num_nodes,
                 const int & i);

// ======================================================================

int main(int argc, char *argv[])
{

  if (argc != 2) {
    cerr << "Error in design: expect exactly one command line variable "
         << "(number of B-spline control points defining nozzle area)"
         << endl;
    throw(-1);
  } else {
    num_design = atoi(argv[1]);
    cout << "Running design with " << num_design << " design vars." << endl;
  }
  
  // define the target area
  InnerProdVector x_coord(nodes, 0.0);
  InnerProdVector y_coord(nodes, 0.0);
  InnerProdVector area(nodes, 0.0);
  for (int i = 0; i < nodes; i++) {
    // create uniform spaced x coordinates
    x_coord(i) = MeshCoord(length, nodes, i);
    area(i) = TargetNozzleArea(x_coord(i));
    y_coord(i) = 0.5*(height - area(i)/width);
  }

  // define the Bspline for the nozzle
  nozzle_shape.SetAreaAtEnds(area_left, area_right);

#if 0
  // query Bspline for nodal areas corresponding to each x
  InnerProdVector area = nozzle_shape.Area(x_coord);
#endif

#if 0
  // calculate the y coordinates based on the nodal areas
  for (int i = 0; i < nodes; i++)
  	y_coord(i) = 0.5*(height - (area(i)/width));  
#endif
  
  // define CSM nodal degrees of freedom
  InnerProdVector BCtype(3*nodes, 0.0);
  InnerProdVector BCval(3*nodes, 0.0);
  for (int i=0; i<nodes; i++) {
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
  BCtype(3*nodes-3) = 0;
  BCtype(3*nodes-2) = 0;
  BCtype(3*nodes-1) = -1;

  // initialize the solver disciplines
  solver.InitializeCFD(x_coord, area);
  solver.InitializeCSM(x_coord, y_coord, BCtype, BCval, E, thick, width, height);
  int precond_calls = solver.NewtonKrylov(30, tol);
  cout << "Total solver precond calls: " << precond_calls << endl;
  solver.CopyPressIntoTarget();

#if 0
  // uncomment to plot initial pressure and displaced area
  solver.GetTecplot(1.0, 1.0);
  throw(-1);
#endif
  
  solver.SetDesignVars(num_design);

  // run the optimizer
  map<string,string> optns;
  KonaOptimize(userFunc, optns);
  int solver_precond_calls = 0;
  int kona_precond_calls = 0;
    
  cout << "Total solver precond calls: " << solver_precond_calls << endl;
  cout << "Total Kona precond calls:   " << kona_precond_calls << endl;
  cout << "Ratio of precond calls:     " 
       << ( static_cast<double>(kona_precond_calls)
            /static_cast<double>(solver_precond_calls) ) << endl;

  solver.NewtonKrylov(20, tol);
  solver.GetTecplot(1.0, 1.0);
  
  // plot results?
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

double InitNozzleArea(const double & x) {
  // linear nozzle
  return area_left + (area_right - area_left)*x;
}

// ======================================================================

double TargetNozzleArea(const double & x) {
  // cubic polynomial nozzle
  //const double area_mid = 1.5;
  double a = area_left;
  double b = 4.0*area_mid - 5.0*area_left + area_right;
  double c = -4.0*(area_right -2.0*area_left + area_mid);
  double d = 4.0*(area_right - area_left);
  return a + x*(b + x*(c + x*d));
}

// ======================================================================

int userFunc(int request, int leniwrk, int *iwrk, int lendwrk,
	     double *dwrk)
{
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
      num_design_vec = iwrk[0];
      num_state_vec = iwrk[1];
      assert(num_design_vec >= 0);
      assert(num_state_vec >= 0);
      design.resize(num_design_vec);
      for (int i = 0; i < num_design_vec; i++)
        design[i].resize(num_design);
      state.resize(num_state_vec);
      for (int i = 0; i < num_state_vec; i++)
        state[i].resize(num_var);
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
    case kona::axpby_c: {
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
    case kona::innerprod_c: {
      break;
    }
    case kona::eval_obj: {// evaluate the objective
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // THIS NEEDS ATTENTION
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      int i = iwrk[0];
      int j = iwrk[1];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= -1) && (j < num_state_vec));
      nozzle_shape.SetCoeff(design[i]);
      solver.UpdateFromNozzle();
      if (j == -1) {
        // need to solve for the state first
        solver.SetInitialCondition();
        iwrk[0] = solver.NewtonKrylov(100, tol);
      } else {
        iwrk[0] = 0; // no precondition calls
        solver.set_u(state[j]);
        solver.UpdateFromNozzle();
      }
      dwrk[0] = solver.CalcInverseDesign();
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
      nozzle_shape.SetCoeff(design[i]);	 // update nozzle design
      solver.set_u(state[j]);	 // set solver state vars at which residual is calculated
      solver.UpdateDisciplineStates();
      solver.UpdateFromNozzle();	// cascade the design update into the solver
      solver.CalcResidual();
      state[k] = solver.get_res();
      break;
    }
    case kona::eval_ceq: {
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
      nozzle_shape.SetCoeff(design[i]);
      solver.set_u(state[j]);
      solver.UpdateDisciplineStates();
      solver.UpdateFromNozzle();
      solver.AeroStructDesignProduct(design[k], state[m]);
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
      nozzle_shape.SetCoeff(design[i]);
      solver.set_u(state[j]);
      solver.UpdateDisciplineStates();
      solver.UpdateFromNozzle();
      kona::MatrixVectorProduct<InnerProdVector>* 
          mat_vec = new AeroStructProduct(&solver);  
      (*mat_vec)(state[k], state[m]);
      delete mat_vec;
      break;
    }
    case kona::tjacvec_d: {// apply design component of Jacobian to adj
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // AeroStructDesignTransProduct() needs review!
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      assert((m >= 0) && (m < num_design_vec));
      nozzle_shape.SetCoeff(design[i]);
      solver.set_u(state[j]);
      solver.UpdateDisciplineStates();
      solver.UpdateFromNozzle();
      solver.AeroStructDesignTransProduct(state[k], design[m]);
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
      nozzle_shape.SetCoeff(design[i]);
      solver.set_u(state[j]);
      solver.UpdateDisciplineStates();
      solver.UpdateFromNozzle();
      kona::MatrixVectorProduct<InnerProdVector>* 
          mat_vec = new AeroStructTransposeProduct(&solver);
      (*mat_vec)(state[k], state[m]);
      delete mat_vec;
      break;
    }
    case kona::eval_precond: {// build the preconditioner if necessary
      int i = iwrk[0];
      int j = iwrk[1];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      nozzle_shape.SetCoeff(design[i]);
      solver.set_u(state[j]);
      solver.UpdateDisciplineStates();
      solver.UpdateFromNozzle();
      solver.BuildAndFactorPreconditioner();
      break;
    }
    case kona::precond_s: {// apply primal preconditioner to iwrk[2]
      // recall; iwrk[0], iwrk[1] denote where preconditioner is
      // evaluated and in this case, they are not needed
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      int m = iwrk[3];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      assert((m >= 0) && (m < num_state_vec));
      kona::Preconditioner<InnerProdVector>* 
          mat_vec = new AeroStructPrecond((&solver));
      (*mat_vec)(state[k], state[m]);
      delete mat_vec;
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
      kona::Preconditioner<InnerProdVector>* 
          mat_vec = new AeroStructTransposePrecond((&solver));
      (*mat_vec)(state[k], state[m]);
      delete mat_vec;
      iwrk[0] = 1; // one preconditioner application
      break;
    }
    case kona::ceqjac_d: {
      break;
    }
    case kona::ceqjac_s: {
      break;
    }
    case kona::tceqjac_d: {
      break;
    }
    case kona::tceqjac_s: {
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
      nozzle_shape.SetCoeff(design[i]);
      solver.set_u(state[j]);
      solver.UpdateDisciplineStates();
      solver.UpdateFromNozzle();
      solver.CalcInverseDesigndJdQ(state[k]);
      //cout << "dJdQ.Norm2() = " << state[k].Norm2() << endl;
      break;
    }
    case kona::initdesign: {// initialize the design variables
      int i = iwrk[0];
      assert((i >= 0) && (i < num_design_vec));
      //design[i] = 0.0; // all coefficients set to zero
      
      // in case the nozzle has not been initiated
      design[i] = 1.0;
      nozzle_shape.SetCoeff(design[i]);
      // fit a b-spline nozzle to a given shape
      InnerProdVector x_coord(nodes, 0.0), area(nodes, 0.0);
      for (int j = 0; j < nodes; j++) {
        x_coord(j) = MeshCoord(length, nodes, j);
        area(j) = InitNozzleArea(x_coord(j)/length);
      }
      nozzle_shape.FitNozzle(x_coord, area);
      nozzle_shape.GetCoeff(design[i]);
      solver.UpdateFromNozzle();

      cout << "kona::initdesign design coeff = ";
      for (int n = 0; n < num_design; n++)
        cout << design[i](n) << " ";
      cout << endl;
      
      break;
    }
    case kona::solve: { // solve the primal equations
      int i = iwrk[0];
      int j = iwrk[1];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      nozzle_shape.SetCoeff(design[i]);
      solver.UpdateFromNozzle();
      //solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
      //solver.GetTecplot(1.0, 1.0);
      //solver.InitialCondition(rho_R, rho_u_R, e_R);
      solver.SetInitialCondition();
      iwrk[0] = solver.NewtonKrylov(20, tol);
      state[j] = solver.get_u();
      break;
    }
    case kona::adjsolve: {// solve the adjoint equations
      int i = iwrk[0];
      int j = iwrk[1];
      int k = iwrk[2];
      assert((i >= 0) && (i < num_design_vec));
      assert((j >= 0) && (j < num_state_vec));
      assert((k >= 0) && (k < num_state_vec));
      nozzle_shape.SetCoeff(design[i]);
      solver.set_u(state[j]);
      solver.UpdateDisciplineStates();
      solver.UpdateFromNozzle();
      InnerProdVector dJdu(num_var, 0.0);
      solver.CalcInverseDesigndJdQ(dJdu);
      dJdu *= -1.0;
      iwrk[0] = solver.SolveAdjoint(100, adj_tol, dJdu, state[k]);
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
      nozzle_shape.SetCoeff(design[i]);

      // uncomment to list B-spline coefficients
      cout << "kona::info current b-spline coeffs = ";
      for (int n = 0; n < num_design; n++)
        cout << design[i](n) << " ";
      cout << endl;

      solver.UpdateFromNozzle();
      solver.set_u(state[j]);
      solver.UpdateDisciplineStates();
      std::string filename = "BFGS_inner_iter_$num.dat";
      solver.GetTecplot(1.0, 1.0);
      
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
