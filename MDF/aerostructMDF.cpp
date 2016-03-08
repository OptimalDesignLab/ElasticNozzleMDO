/**
 * \file optimizer.cpp
 * \brief main binary for aero-structural MDA optimization
 * \author Jason Hicken <jason.hicken@gmail.com>, Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

#include "./aerostructMDF.hpp"

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

double InitNozzleArea(const double & x) {
  // linear nozzle
  return area_left + (area_right - area_left)*x;
}

double TargetNozzleArea(const double & x) {
#if 1
  // cubic polynomial nozzle
  double a = area_left;
  double b = 4.0*area_mid - 5.0*area_left + area_right;
  double c = -4.0*(area_right -2.0*area_left + area_mid);
  double d = 4.0*(area_right - area_left);
  return a + x*(b + x*(c + x*d));
#else
  // cosine series nozzle
  double a = 0.25*(area_left + area_right) + 0.5*area_mid;
  double b = 0.5*(area_left - area_right);
  double c = 0.25*(area_left + area_right) - 0.5*area_mid;
  return a + b*cos(pi*x) + c*cos(2.0*pi*x);
#endif
}

// ======================================================================

void init_mda(int py_num_design)
{
  // set number of design variables
  assert(py_num_design > 0);
  num_design = py_num_design;
  cout << "Running design with " << num_design << " design vars." << endl;

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
  cout << "Cost of one MDA eval: " << precond_calls << endl;

#if 0
  // uncomment to plot initial pressure and displaced area
  solver.GetTecplot(1.0, 1.0);
  throw(-1);
#endif

  solver.SetDesignVars(num_design);
}

void alloc_mem(int py_num_design_vec, int py_num_state_vec)
{
  assert(py_num_design_vec >= 0);
  assert(py_num_state_vec >= 0);
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
  num_design_vec = py_num_design_vec;
  design.resize(num_design_vec);
  for (int i = 0; i < num_design_vec; i++)
    design[i].resize(num_design);
  state.resize(num_state_vec);
  for (int i = 0; i < num_state_vec; i++)
    state[i].resize(num_var);
}

void init_design(int store_here)
{
  assert((store_here >= 0) && (store_here < num_design_vec));
  //design[i] = 0.0; // all coefficients set to zero

  // in case the nozzle has not been initiated
  design[store_here] = 1.0;
  nozzle_shape.SetCoeff(design[store_here]);
  // fit a b-spline nozzle to a given shape
  InnerProdVector x_coord(nodes, 0.0), area(nodes, 0.0);
  for (int j = 0; j < nodes; j++) {
    x_coord(j) = MeshCoord(length, nodes, j);
    area(j) = InitNozzleArea(x_coord(j)/length);
  }
  nozzle_shape.FitNozzle(x_coord, area);
  nozzle_shape.GetCoeff(design[store_here]);
  solver.UpdateFromNozzle();

  cout << "kona::initdesign design coeff = ";
  for (int n = 0; n < num_design; n++)
    cout << design[store_here](n) << " ";
  cout << endl;
}

void info_dump(int at_design, int at_state, int adjoint, int iter)
{
  nozzle_shape.SetCoeff(design[at_design]);
  // uncomment to list B-spline coefficients
  cout << "kona::info current b-spline coeffs = ";
  for (int n = 0; n < num_design; n++)
    cout << design[at_design](n) << " ";
  cout << endl;
  solver.UpdateFromNozzle();
  solver.set_u(state[at_state]);
  solver.UpdateDisciplineStates();
  std::string filename = "BFGS_inner_iter_$num.dat";
  solver.GetTecplot(1.0, 1.0);
}

// ======================================================================

void axpby_d(int i, double scalj, int j, double scalk, int k)
{
  assert((i >= 0) && (i < num_design_vec));
  assert(j < num_design_vec);
  assert(k < num_design_vec);

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
}

void times_vector_d(int i, int j)
{
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_design_vec));
  int n;
  for (n = 0; n < num_design; n++)
    design[i](n) *= design[j](n);
}

void exp_d(int i)
{
  assert((i >= 0) && (i < num_design_vec));
  int n;
  for (n = 0; n < num_design; n++)
    design[i](n) = exp(design[i](n));
}

void log_d(int i)
{
  assert((i >= 0) && (i < num_design_vec));
  int n;
  for (n = 0; n < num_design; n++)
    design[i](n) = log(design[i](n));
}

void pow_d(int i, double p)
{
  assert((i >= 0) && (i < num_design_vec));
  int n;
  for (n = 0; n < num_design; n++)
    design[i](n) = pow(design[i](n), p);
}

double inner_d(int i, int j)
{
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_design_vec));
  return InnerProd(design[i], design[j]);
}

// ======================================================================

void axpby_s(int i, double scalj, int j, double scalk, int k)
{
  assert((i >= 0) && (i < num_state_vec));
  assert(j < num_state_vec);
  assert(k < num_state_vec);

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
}

void times_vector_s(int i, int j)
{
  assert((i >= 0) && (i < num_state_vec));
  assert((j >= 0) && (j < num_state_vec));
  int n;
  for (n = 0; n < num_state; n++)
    state[i](n) *= state[j](n);
}

void exp_s(int i)
{
  assert((i >= 0) && (i < num_state_vec));
  int n;
  for (n = 0; n < num_state; n++)
    state[i](n) = exp(state[i](n));
}

void log_s(int i)
{
  assert((i >= 0) && (i < num_state_vec));
  int n;
  for (n = 0; n < num_state; n++)
    state[i](n) = log(state[i](n));
}

void pow_s(int i, double p)
{
  assert((i >= 0) && (i < num_state_vec));
  int n;
  for (n = 0; n < num_state; n++)
    state[i](n) = pow(state[i](n), p);
}

double inner_s(int i, int j)
{
  assert((i >= 0) && (i < num_state_vec));
  assert((j >= 0) && (j < num_state_vec));
  return InnerProd(state[i], state[j]);
}

// ======================================================================

boost::python::tuple eval_f(int at_design, int at_state)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= -1) && (at_state < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.UpdateFromNozzle();
  int cost = 0;
  if (at_state == -1) {
    // need to solve for the state first
    solver.SetInitialCondition();
    cost = solver.NewtonKrylov(100, tol);
  } else {
    solver.set_u(state[at_state]);
    solver.UpdateFromNozzle();
  }
  return boost::python::make_tuple(solver.CalcInverseDesign(), cost);
}

void eval_dfdx(int at_design, int at_state, int store_here)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((store_here >= 0) && (store_here < num_design_vec));
  design[store_here] = 0.0;
}

void eval_dfdw(int at_design, int at_state, int store_here)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((store_here >= 0) && (store_here < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.set_u(state[at_state]);
  solver.UpdateDisciplineStates();
  solver.UpdateFromNozzle();
  solver.CalcInverseDesigndJdQ(state[store_here]);
}

// ======================================================================

void eval_r(int at_design, int at_state, int store_here)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((store_here >= 0) && (store_here < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);	 // update nozzle design
  solver.set_u(state[at_state]);	 // set solver state vars at which residual is calculated
  solver.UpdateDisciplineStates();
  solver.UpdateFromNozzle();	// cascade the design update into the solver
  solver.CalcResidual();
  state[store_here] = solver.get_res();
}

void mult_drdx(int at_design, int at_state, int in_vec, int out_vec)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((in_vec >= 0) && (in_vec < num_design_vec));
  assert((out_vec >= 0) && (out_vec < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.set_u(state[at_state]);
  solver.UpdateDisciplineStates();
  solver.UpdateFromNozzle();
  solver.AeroStructDesignProduct(design[in_vec], state[out_vec]);
}

void mult_drdx_t(int at_design, int at_state, int in_vec, int out_vec)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((in_vec >= 0) && (in_vec < num_state_vec));
  assert((out_vec >= 0) && (out_vec < num_design_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.set_u(state[at_state]);
  solver.UpdateDisciplineStates();
  solver.UpdateFromNozzle();
  solver.AeroStructDesignTransProduct(state[in_vec], design[out_vec]);
}

void mult_drdw(int at_design, int at_state, int in_vec, int out_vec)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((in_vec >= 0) && (in_vec < num_state_vec));
  assert((out_vec >= 0) && (out_vec < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.set_u(state[at_state]);
  solver.UpdateDisciplineStates();
  solver.UpdateFromNozzle();
  kona::MatrixVectorProduct<InnerProdVector>*
      mat_vec = new AeroStructProduct(&solver);
  (*mat_vec)(state[in_vec], state[out_vec]);
  delete mat_vec;
}

void mult_drdw_t(int at_design, int at_state, int in_vec, int out_vec)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((in_vec >= 0) && (in_vec < num_state_vec));
  assert((out_vec >= 0) && (out_vec < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.set_u(state[at_state]);
  solver.UpdateDisciplineStates();
  solver.UpdateFromNozzle();
  kona::MatrixVectorProduct<InnerProdVector>*
      mat_vec = new AeroStructTransposeProduct(&solver);
  (*mat_vec)(state[in_vec], state[out_vec]);
  delete mat_vec;
}

// ======================================================================

void factor_precond(int at_design, int at_state)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.set_u(state[at_state]);
  solver.UpdateDisciplineStates();
  solver.UpdateFromNozzle();
  solver.BuildAndFactorPreconditioner();
}

int apply_precond(int at_design, int at_state, int in_vec, int out_vec)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((in_vec >= 0) && (in_vec < num_state_vec));
  assert((out_vec >= 0) && (out_vec < num_state_vec));
  kona::Preconditioner<InnerProdVector>*
      mat_vec = new AeroStructPrecond((&solver));
  (*mat_vec)(state[in_vec], state[out_vec]);
  delete mat_vec;
  return 1;
}

int apply_precond_t(int at_design, int at_state, int in_vec, int out_vec)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((in_vec >= 0) && (in_vec < num_state_vec));
  assert((out_vec >= 0) && (out_vec < num_state_vec));
  kona::Preconditioner<InnerProdVector>*
      mat_vec = new AeroStructTransposePrecond((&solver));
  (*mat_vec)(state[in_vec], state[out_vec]);
  delete mat_vec;
}

// ======================================================================

int solve_nonlinear(int at_design, int store_here)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((store_here >= 0) && (store_here < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.UpdateFromNozzle();
  //solver.set_area(nozzle_shape.Area(solver.get_x_coord()));
  //solver.GetTecplot(1.0, 1.0);
  //solver.InitialCondition(rho_R, rho_u_R, e_R);
  solver.SetInitialCondition();
  int cost = solver.NewtonKrylov(20, tol);
  state[store_here] = solver.get_u();
  return cost;
}

int solve_linear(int at_design, int at_state, int rhs, int result)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((rhs >= 0) && (rhs < num_state_vec));
  assert((result >= 0) && (result < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.set_u(state[at_state]);
  solver.UpdateDisciplineStates();
  solver.UpdateFromNozzle();
  return solver.SolveLinearized(10000, adj_tol, state[rhs], state[result]);
}

int solve_adjoint(int at_design, int at_state, int rhs, int result)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= 0) && (at_state < num_state_vec));
  assert((rhs >= 0) && (rhs < num_state_vec));
  assert((result >= 0) && (result < num_state_vec));
  nozzle_shape.SetCoeff(design[at_design]);
  solver.set_u(state[at_state]);
  solver.UpdateDisciplineStates();
  solver.UpdateFromNozzle();
  return solver.SolveAdjoint(10000, adj_tol, state[rhs], state[result]);
}

// ======================================================================

BOOST_PYTHON_MODULE(aerostructMDF)
{
  using namespace boost::python;

  def("init_mda", init_mda);
  def("alloc_mem", alloc_mem);
  def("init_design", init_design);
  def("info_dump", info_dump);

  def("axpby_d", axpby_d);
  def("times_vector_d", times_vector_d);
  def("exp_d", exp_d);
  def("log_d", log_d);
  def("pow_d", pow_d);
  def("inner_d", inner_d);

  def("axpby_s", axpby_s);
  def("times_vector_s", times_vector_s);
  def("exp_s", exp_s);
  def("log_s", log_s);
  def("pow_s", pow_s);
  def("inner_s", inner_s);

  def("eval_f", eval_f);
  def("eval_dfdx", eval_dfdx);
  def("eval_dfdw", eval_dfdw);

  def("eval_r", eval_r);
  def("mult_drdx", mult_drdx);
  def("mult_drdx_t", mult_drdx_t);
  def("mult_drdw", mult_drdw);
  def("mult_drdw_t", mult_drdw_t);

  def("factor_precond", factor_precond);
  def("apply_precond", apply_precond);
  def("apply_precond_t", apply_precond_t);

  def("solve_nonlinear", solve_nonlinear);
  def("solve_linear", solve_linear);
  def("solve_adjoint", solve_adjoint);
}
