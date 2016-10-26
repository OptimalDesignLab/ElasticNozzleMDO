
#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include <Python.h>

#include <boost/math/constants/constants.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <boost/python.hpp>

#include <pyublas/numpy.hpp>

#include "../Quasi1DEuler/inner_prod_vector.hpp"
#include "../Quasi1DEuler/exact_solution.hpp"
#include "../Quasi1DEuler/nozzle.hpp"
#include "../Quasi1DEuler/quasi_1d_euler.hpp"
#include "../LECSM/lecsm.hpp"
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

// global variables...but only to this file
static int num_design_vec = -1; // number of design vectors
static int num_state_vec = -1;  // number of state vectors
static int num_dual_vec = -1;   // number of dual vectors
static vector<InnerProdVector> design;    // design vectors
static vector<InnerProdVector> state;     // state vectors
static vector<InnerProdVector> dual;      // dual vectors

// design parameters
static int num_design; // total number of design vars (control points + coupling)
static int num_bspline; // number of b-spline control points

// some vector sizes
static int nodes;
static int num_dis_var; // number of states in a discipline
static int num_var; // number of states total
static int num_ceq; // number of equality constraints

// used for initial conditions and scaling
static double rho_R, rho_u_R, e_R;
static double p_ref;

static InnerProdVector press_stag; //(nodes, kPressStag);
static InnerProdVector press_targ; //(nodes, 0.0);
static LECSM csm_solver; //nodes
static Quasi1DEuler cfd_solver; //nodes, order
static BsplineNozzle nozzle_shape;

double MeshCoord(const double & length, const int & num_nodes,
                 const int & i);

double TargetNozzleArea(const double & x);

double InitNozzleArea(const double & x);

int FindTargPress(const InnerProdVector & x_coord,
                  const InnerProdVector & BCtype,
                  const InnerProdVector & BCval,
                  InnerProdVector & targ_press);

void InitCFDSolver(const InnerProdVector & x_coord,
                   const InnerProdVector & press_targ);

void InitCSMSolver(const InnerProdVector & x_coord,
                   const InnerProdVector & y_coord,
                   const InnerProdVector & BCtype,
                   const InnerProdVector & BCval);

void CalcYCoords(const InnerProdVector & area, InnerProdVector & y_coord);

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

/* Kona control functions */
/* ====================== */

// general
void init_mda(int py_num_design, int py_nodes);
void alloc_design(int py_num_design_vec);
void alloc_state(int py_num_state_vec);
void alloc_dual(int py_num_dual_vec);
void init_design(int store_here);
void info_dump(int at_design, int at_state, int adjoint, int iter);

// io functions
void set_design_data(int idx, pyublas::numpy_vector<double> data);
void set_state_data(int idx, pyublas::numpy_vector<double> data);
void set_dual_data(int idx, pyublas::numpy_vector<double> data);
pyublas::numpy_vector<double> get_design_data(int idx);
pyublas::numpy_vector<double> get_state_data(int idx);
pyublas::numpy_vector<double> get_dual_data(int idx);

// design vec linalg
void axpby_d(int i, double scalj, int j, double scalk, int k);
void times_vector_d(int i, int j);
void exp_d(int i);
void log_d(int i);
void pow_d(int i, double p);
double inner_d(int i, int j);

// state vec linalg
void axpby_s(int i, double scalj, int j, double scalk, int k);
void times_vector_s(int i, int j);
void exp_s(int i);
void log_s(int i);
void pow_s(int i, double p);
double inner_s(int i, int j);

// dual vec linalg
void axpby_c(int i, double scalj, int j, double scalk, int k);
void times_vector_c(int i, int j);
void exp_c(int i);
void log_c(int i);
void pow_c(int i, double p);
double inner_c(int i, int j);

// idf vector ops
void zero_targ_state(int at_design);
void zero_real_design(int at_design);
void copy_dual_to_targ_state(int take_from, int copy_to);
void copy_targ_state_to_dual(int take_from, int copy_to);

// objective function
boost::python::tuple eval_f(int at_design, int at_state);
void eval_dfdx(int at_design, int at_state, int store_here);
void eval_dfdw(int at_design, int at_state, int store_here);

// residual
void eval_r(int at_design, int at_state, int store_here);
void mult_drdx(int at_design, int at_state, int in_vec, int out_vec);
void mult_drdx_t(int at_design, int at_state, int in_vec, int out_vec);
void mult_drdw(int at_design, int at_state, int in_vec, int out_vec);
void mult_drdw_t(int at_design, int at_state, int in_vec, int out_vec);

// constraint
void eval_c(int at_design, int at_state, int store_here);
void mult_dcdx(int at_design, int at_state, int in_vec, int out_vec);
void mult_dcdx_t(int at_design, int at_state, int in_vec, int out_vec);
void mult_dcdw(int at_design, int at_state, int in_vec, int out_vec);
void mult_dcdw_t(int at_design, int at_state, int in_vec, int out_vec);

// precond
void factor_precond(int at_design, int at_state);
int apply_precond(int at_design, int at_state, int in_vec, int out_vec);
int apply_precond_t(int at_design, int at_state, int in_vec, int out_vec);

// solutions
int solve_nonlinear(int at_design, int store_here);
int solve_linear(int at_design, int at_state, int rhs, int result, double rel_tol);
int solve_adjoint(int at_design, int at_state, int rhs, int result, double rel_tol);
