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
#include <math.h>

#include <Python.h>

#include <boost/math/constants/constants.hpp>
#include <boost/python.hpp>

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
static int nodes;
static int num_var;
static int num_design;

static BsplineNozzle nozzle_shape;
static AeroStructMDA solver;

// function declarations
double InitNozzleArea(const double & x);

double TargetNozzleArea(const double & x);

double MeshCoord(const double & length, const int & num_nodes,
                 const int & i);

/* Kona control functions */
/* ====================== */

// general
void init_mda(int py_num_design, int py_nodes);
void alloc_design(int py_num_design_vec);
void alloc_state(int py_num_state_vec);
void init_design(int store_here);
void info_dump(int at_design, int at_state, int adjoint, int iter);

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

// precond
void factor_precond(int at_design, int at_state);
int apply_precond(int at_design, int at_state, int in_vec, int out_vec);
int apply_precond_t(int at_design, int at_state, int in_vec, int out_vec);

// solutions
int solve_nonlinear(int at_design, int store_here);
int solve_linear(int at_design, int at_state, int rhs, int result);
int solve_adjoint(int at_design, int at_state, int rhs, int result);
