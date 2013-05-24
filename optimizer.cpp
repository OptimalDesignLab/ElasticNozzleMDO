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

#include "./aerostruct.hpp"

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

// domain parameters
static const double length = 1.0;
static const double x_min = 0.0;
static const double x_max = x_min + length;

// discretization parameters
static const int nodes = 201;
static const int num_var = 3*nodes;
static const int order = 3;

static AeroStructMDA solver(nodes, order);
static BsplineNozzle nozzle_shape;

// forward declerations
int userFunc(int request, int leniwrk, int *iwrk, int lendwrk,
	     double *dwrk);
double MeshCoord(const double & length, const int & num_nodes,
                 const int & i);

// ======================================================================

int main(int argc, char *argv[]) 
{
	// create uniform spaced x coordinates
	InnerProdVector x_coord(nodes, 0.0);
  for (int i = 0; i < nodes; i++)
    x_coord(i) = MeshCoord(length, nodes, i);

  // define Bspline nozzle (???)

  // query Bspline for y coordinates corresponding to each x

  // calculate nodal areas based on y coordinates

  // initialize the CFD discipline

  // define material properties

  // define CSM nodal degrees of freedom

  // initialize the CSM discipline

  // KonaOptimize (magic!)

  // plot results
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
	  case kona::eval_obj: {// evaluate the objective
	    int i = iwrk[0];
	    int j = iwrk[1];      
	    assert((i >= 0) && (i < num_design_vec));
	    assert((j >= -1) && (j < num_state_vec));
	    nozzle_shape.SetCoeff(design[i]);
	    solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
	    if (j == -1) {
	      // need to solve for the state first
	solver.InitialCondition(rho_R, rho_u_R, e_R);
	      iwrk[0] = solver.NewtonKrylov(100, 1.e-6);
	    } else {
	      iwrk[0] = 0; // no precondition calls
	      solver.set_q(state[j]);
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
	    nozzle_shape.SetCoeff(design[i]);
	    solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
	    solver.set_solution_vec(state[j]);
	    solver.CalcResidual();
	    state[k] = solver.get_res();
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
	    solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
	    solver.set_q(state[j]);

	    InnerProdVector dArea(num_design, 0.0);
	    dArea = nozzle_shape.AreaForwardDerivative(solver.get_x_coord(),
	                                               design[k]);
	    solver.JacobianAreaProduct(dArea, state[m]);
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
	    solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
	    solver.set_q(state[j]);
	    solver.JacobianStateProduct(state[k], state[m]);
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
	    nozzle_shape.SetCoeff(design[i]);
	    solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
	    solver.set_q(state[j]);

	    InnerProdVector dArea(nodes, 0.0);
	    solver.JacobianTransposedAreaProduct(state[k], dArea);
	    design[m] = nozzle_shape.AreaReverseDerivative(
	        solver.get_x_coord(), dArea);
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
	    solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
	    solver.set_q(state[j]);
	    solver.JacobianTransposedStateProduct(state[k], state[m]);
	    break;
	  }
	  case kona::eval_precond: {// build the preconditioner if necessary
	    int i = iwrk[0];
	    int j = iwrk[1];
	    assert((i >= 0) && (i < num_design_vec));
	    assert((j >= 0) && (j < num_state_vec));
	    nozzle_shape.SetCoeff(design[i]);
	    solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
	    solver.set_q(state[j]);
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
	    solver.Precondition(state[k], state[m]);
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
	    solver.PreconditionTransposed(state[k], state[m]);
	    iwrk[0] = 1; // one preconditioner application
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
	    solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
	    solver.set_q(state[j]);
	    solver.CalcInverseDesigndJdQ(state[k]);
	    //cout << "dJdQ.Norm2() = " << state[k].Norm2() << endl;
	    break;
	  }
	}
}	