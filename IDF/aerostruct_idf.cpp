#include "./aerostruct_idf.hpp"

// ======================================================================
// HELPER FUNCTIONS
// ======================================================================

double MeshCoord(const double & length, const int & num_nodes,
                 const int & i) {
  double xi = static_cast<double>(i)/static_cast<double>(num_nodes-1);
  // uniform spacing
  return length*xi;
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

double InitNozzleArea(const double & x) {
  // linear nozzle
  return area_left + (area_right - area_left)*x;
}

int FindTargPress(const InnerProdVector & x_coord,
                  const InnerProdVector & BCtype,
                  const InnerProdVector & BCval,
                  InnerProdVector & targ_press) {
  // define the target nozzle area and corresponding y-coord
  InnerProdVector area(nodes, 0.0), y_coord(nodes, 0.0);
  for (int i = 0; i < nodes; i++) {
    area(i) = TargetNozzleArea(x_coord(i));
    y_coord(i) = 0.5*(height - area(i)/width);
  }
  AeroStructMDA asmda(nodes, order);
  asmda.InitializeCFD(x_coord, area);
  asmda.InitializeCSM(x_coord, y_coord, BCtype, BCval, E, thick, width, height);
  int precond_calls = asmda.NewtonKrylov(30, tol);
  //asmda.GetTecplot(1.0, 1.0);
  //throw(-1);
  targ_press = asmda.get_press();
  return precond_calls;
}

void InitCFDSolver(const InnerProdVector & x_coord,
                   const InnerProdVector & press_t) {
  // define reference and boundary conditions
  double rho, rho_u, e;
  CalcFlowExact(kGamma, kRGas, kAreaStar, area_left, true,
                kTempStag, kPressStag, rho, rho_u, e);
  double rho_ref = rho;
  double press = (kGamma - 1.0)*(e - 0.5*rho_u*rho_u/rho);
  p_ref = press; // defines static variable
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
  cfd_solver.set_press_targ(press_t);
}

void InitCSMSolver(const InnerProdVector & x_coord,
                   const InnerProdVector & y_coord,
                   const InnerProdVector & BCtype,
                   const InnerProdVector & BCval) {
  // set material properties
  csm_solver.set_material(E, thick, width, height);
  // create the CSM mesh
  csm_solver.GenerateMesh(x_coord, y_coord);
  // set boundary conditions
  csm_solver.SetBoundaryConds(BCtype, BCval);
}

void CalcYCoords(const InnerProdVector & area, InnerProdVector & y_coord) {
  assert(area.size() == nodes);
  assert(y_coord.size() == nodes);
  for (int j = 0; j < nodes; j++)
    y_coord(j) = 0.5*(height - area(j)/width);
}

void GetBsplinePts(const int & i, InnerProdVector & pts) {
  assert((i >= 0) && (i < num_design_vec));
  assert(pts.size() == num_bspline);
  for (int j = 0; j < num_bspline; j++)
    pts(j) = design[i](j);
}

void GetCouplingPress(const int & i, InnerProdVector & press) {
  assert((i >= 0) && (i < num_design_vec));
  assert(press.size() == nodes);
  for (int j = 0; j < nodes; j++)
    press(j) = design[i](num_bspline+j);
}

void GetCouplingArea(const int & i, InnerProdVector & area) {
  assert((i >= 0) && (i < num_design_vec));
  assert(area.size() == nodes);
  for (int j = 0; j < nodes; j++)
    area(j) = design[i](num_bspline + nodes + j);
}

void SetBsplinePts(const int & i, const InnerProdVector & pts) {
  assert((i >= 0) && (i < num_design_vec));
  assert(pts.size() == num_bspline);
  for (int j = 0; j < num_bspline; j++) {
    design[i](j) = pts(j);
  }
}

void SetCouplingPress(const int & i, const InnerProdVector & press) {
  assert((i >= 0) && (i < num_design_vec));
  assert(press.size() == nodes);
  for (int j = 0; j < nodes; j++)
    design[i](num_bspline+j) = press(j);
}

void SetCouplingArea(const int & i, const InnerProdVector & area) {
  assert((i >= 0) && (i < num_design_vec));
  assert(area.size() == nodes);
  for (int j = 0; j < nodes; j++)
    design[i](num_bspline + nodes + j) = area(j);
}

void GetCFDState(const int & i, InnerProdVector & q) {
  assert((i >= 0) && (i < num_state_vec));
  assert(q.size() == num_dis_var);
  for (int j = 0; j < num_dis_var; j++)
    q(j) = state[i](j);
}

void GetCSMState(const int & i, InnerProdVector & u) {
  assert((i >= 0) && (i < num_state_vec));
  assert(u.size() == num_dis_var);
  for (int j = 0; j < num_dis_var; j++)
    u(j) = state[i](num_dis_var+j);
}

void SetCFDState(const int & i, const InnerProdVector & q) {
  assert((i >= 0) && (i < num_state_vec));
  assert(q.size() == num_dis_var);
  for (int j = 0; j < num_dis_var; j++)
    state[i](j) = q(j);
}

void SetCSMState(const int & i, const InnerProdVector & u) {
  assert((i >= 0) && (i < num_state_vec));
  assert(u.size() == num_dis_var);
  for (int j = 0; j < num_dis_var; j++)
    state[i](num_dis_var+j) = u(j);
}

void GetPressCnstr(const int & i, InnerProdVector & ceq_press) {
  assert((i >= 0) && (i < num_dual_vec));
  assert(ceq_press.size() == nodes);
  for (int j = 0; j < nodes; j++)
    ceq_press(j) = dual[i](j);
}

void GetAreaCnstr(const int & i, InnerProdVector & ceq_area) {
  assert((i >= 0) && (i < num_dual_vec));
  assert(ceq_area.size() == nodes);
  for (int j = 0; j < nodes; j++)
    ceq_area(j) = dual[i](nodes + j);
}

void SetPressCnstr(const int & i, const InnerProdVector & ceq_press) {
  assert((i >= 0) && (i < num_dual_vec));
  assert(ceq_press.size() == nodes);
  for (int j = 0; j < nodes; j++)
    dual[i](j) = ceq_press(j);
}

void SetAreaCnstr(const int & i, const InnerProdVector & ceq_area) {
  assert((i >= 0) && (i < num_dual_vec));
  assert(ceq_area.size() == nodes);
  for (int j = 0; j < nodes; j++)
     dual[i](nodes + j) = ceq_area(j);
}

// ======================================================================
// SOLVER AND MEMORY INITIALIZATION
// ======================================================================

void init_mda(int py_num_design, int py_nodes)
{
  assert(py_num_design > 0);
  assert((py_nodes == 61) || (py_nodes = 121));
  nodes = py_nodes;
  num_bspline = py_num_design;
  num_design = num_bspline + 2*nodes;
  num_dis_var = 3*nodes; // number of states in a discipline
  num_var = 2*num_dis_var; // number of states total
  num_ceq = 2*nodes; // number of equality constraints

  cout << "# of real design = " << num_bspline << endl;
  cout << "# of total design = " << num_design << endl;
  cout << "# of cnstr = " << num_ceq << endl;
  cout << "# of nodes  = " << nodes << endl;
  cout << "# of state  = " << num_var << endl;

  press_stag = InnerProdVector(nodes, kPressStag);
  press_targ = InnerProdVector(nodes, 0.0);
  csm_solver = LECSM(nodes); //nodes
  cfd_solver = Quasi1DEuler(nodes, order); //nodes, order

  // set the nodal degrees of freedom for the CSM
  InnerProdVector BCtype(num_dis_var, 0.0);
  InnerProdVector BCval(num_dis_var, 0.0);
  for (int i = 0; i < nodes; i++) {
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
  BCtype(num_dis_var-3) = 0;
  BCtype(num_dis_var-2) = 0;
  BCtype(num_dis_var-1) = -1;

  // define the x- and y-coordinates, and the linearly varying nozzle area
  InnerProdVector x_coord(nodes, 0.0), y_coord(nodes, 0.0), area(nodes, 0.0);
  for (int i = 0; i < nodes; i++) {
    x_coord(i) = MeshCoord(length, nodes, i);
    area(i) = InitNozzleArea(x_coord(i)/length);
    y_coord(i) = 0.5*(height - area(i)/width);
  }

  // define the left and right nozzle areas
  nozzle_shape.SetAreaAtEnds(area_left, area_right);

  // find the target pressure and the number of preconditioner calls for the MDA
  int solver_precond_calls = FindTargPress(x_coord, BCtype, BCval, press_targ);
  cout << "total MDA precond calls = " << solver_precond_calls << endl;

  // set-up the cfd_solver
  cout << "Initializing CDF...";
  InitCFDSolver(x_coord, press_targ);
  cout << "DONE" << endl;
  // set-up the csm_solver
  cout << "Initializing CSM...";
  InitCSMSolver(x_coord, y_coord, BCtype, BCval);
  cout << "DONE" << endl;
}

void alloc_design(int py_num_design_vec)
{
  cout << "Allocating design vectors...";
  if (num_design_vec >= 0) { // free design memory first
    if (design.size() == 0) {
      cerr << "userFunc: "
           << "design array is empty but num_design_vec > 0" << endl;
      throw(-1);
    }
    design.clear();
  }
  num_design_vec = py_num_design_vec;
  assert(num_design_vec >= 0);
  design.resize(num_design_vec);
  for (int i = 0; i < num_design_vec; i++)
    design[i].resize(num_design);
  cout << "DONE" << endl;
}

void alloc_state(int py_num_state_vec)
{
  cout << "Allocating state vectors...";
  if (num_state_vec >= 0) { // free state memory first
    if (state.size() == 0) {
      cerr << "userFunc: "
           << "state array is empty but num_state_vec > 0" << endl;
      throw(-1);
    }
    state.clear();
  }
  num_state_vec = py_num_state_vec;
  assert(num_state_vec >= 0);
  state.resize(num_state_vec);
  for (int i = 0; i < num_state_vec; i++)
    state[i].resize(num_var);
  cout << "DONE" << endl;
}

void alloc_dual(int py_num_dual_vec)
{
  cout << "Allocating dual vectors...";
  if (num_dual_vec >= 0) { // free dual memory first
    if (dual.size() == 0) {
      cerr << "userFunc: "
           << "dual array is empty but num_dual_vec > 0" << endl;
      throw(-1);
    }
    dual.clear();
  }
  num_dual_vec = py_num_dual_vec;
  assert(num_dual_vec >= 0);
  dual.resize(num_dual_vec);
  for (int i = 0; i < num_dual_vec; i++)
    dual[i].resize(num_ceq);
  cout << "DONE" << endl;
}

void init_design(int store_here)
{
  assert((store_here >= 0) && (store_here < num_design_vec));
  // in case the nozzle has not been initiated
  InnerProdVector pts(num_bspline, 0.0), press(nodes, 0.0),
      area(nodes, 0.0);
  pts = 1.0;

  SetBsplinePts(store_here, pts);
  nozzle_shape.SetCoeff(pts);

  // fit a b-spline nozzle to a given shape
  InnerProdVector x_coord(nodes, 0.0);
  for (int j = 0; j < nodes; j++) {
    x_coord(j) = MeshCoord(length, nodes, j);
    area(j) = InitNozzleArea(x_coord(j)/length);
  }
  nozzle_shape.FitNozzle(x_coord, area);
  nozzle_shape.GetCoeff(pts);

  SetBsplinePts(store_here, pts);
  SetCouplingArea(store_here, area);

  // Solve for pressure based on initial area (is there a better way?)
  cfd_solver.InitialCondition(rho_R, rho_u_R, e_R);
  cfd_solver.set_area(area);
  int cost = cfd_solver.NewtonKrylov(100, 1.e-6);
  cfd_solver.WriteTecplot(1.0, 1.0, "init_pressure_area.dat");
  press = cfd_solver.get_press();

  SetCouplingPress(store_here, press);
  cout << "kona::initdesign design coeff = ";
  for (int n = 0; n < num_bspline; n++)
    cout << pts(n) << " ";
  cout << endl;
}

void info_dump(int at_design, int at_state, int at_adjoint, int iter)
{
  InnerProdVector pts(num_bspline, 0.0), area(nodes, 0.0), press(nodes, 0.0),
      q(num_dis_var, 0.0), u(num_dis_var, 0.0), y_coords(nodes, 0.0);
  GetBsplinePts(at_design, pts);
  nozzle_shape.SetCoeff(pts);

  GetCouplingArea(at_design, area);
  cfd_solver.set_area(area);
  GetCFDState(at_state, q);
  cfd_solver.set_q(q);
  cfd_solver.CalcAuxiliaryVariables(q);
  cfd_solver.WriteTecplot(1.0, 1.0);

  GetCouplingPress(at_design, press);
  GetCSMState(at_state, u);
  area = nozzle_shape.Area(cfd_solver.get_x_coord());
  CalcYCoords(area, y_coords);
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  csm_solver.set_u(u);
  csm_solver.CalcCoordsAndArea();

  cfd_solver.set_area(csm_solver.get_area());
  cfd_solver.WriteTecplot(1.0, 1.0, "csm_area.dat");

  // uncomment to list B-spline coefficients
  // cout << "before kona::info set_area: design coeff = ";
  // for (int n = 0; n < num_bspline; n++)
  //   cout << design[at_design](n) << " ";
  // cout << endl;

#if 0
  string filename("flow_at_opt_iter");
  std::stringstream ss;
  ss << opt_iter;
  filename.append(ss.str());
  filename.append(".dat");
  cfd_solver.WriteTecplot(1.0, 1.0, filename);
  opt_iter++;
#endif
}

// ======================================================================
// DATA I/O
// ======================================================================

void set_design_data(int idx, pyublas::numpy_vector<double> data)
{
  assert(data.as_ublas().size() == num_design);
  design[idx] = InnerProdVector(data.as_ublas());
}

pyublas::numpy_vector<double> get_design_data(int idx)
{
  assert(design[idx].size() == num_design);
  return pyublas::numpy_vector<double>(
    static_cast<ublas::vector<double> >(design[idx]));
}

void set_state_data(int idx, pyublas::numpy_vector<double> data)
{
  assert(data.as_ublas().size() == num_var);
  state[idx] = InnerProdVector(data.as_ublas());
}

pyublas::numpy_vector<double> get_state_data(int idx)
{
  assert(state[idx].size() == num_var);
  return pyublas::numpy_vector<double>(
    static_cast<ublas::vector<double> >(state[idx]));
}

void set_dual_data(int idx, pyublas::numpy_vector<double> data)
{
  assert(data.as_ublas().size() == num_ceq);
  dual[idx] = InnerProdVector(data.as_ublas());
}

pyublas::numpy_vector<double> get_dual_data(int idx)
{
  assert(dual[idx].size() == num_ceq);
  return pyublas::numpy_vector<double>(
    static_cast<ublas::vector<double> >(dual[idx]));
}

// ======================================================================
// LINEAR ALGEBRA
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
    for (n = 0; n < num_var; n++)
        state[i](n) *= state[j](n);
}

void exp_s(int i)
{
    assert((i >= 0) && (i < num_state_vec));
    int n;
    for (n = 0; n < num_var; n++)
        state[i](n) = exp(state[i](n));
}

void log_s(int i)
{
    assert((i >= 0) && (i < num_state_vec));
    int n;
    for (n = 0; n < num_var; n++)
        state[i](n) = log(state[i](n));
}

void pow_s(int i, double p)
{
    assert((i >= 0) && (i < num_state_vec));
    int n;
    for (n = 0; n < num_var; n++)
        state[i](n) = pow(state[i](n), p);
}

double inner_s(int i, int j)
{
    assert((i >= 0) && (i < num_state_vec));
    assert((j >= 0) && (j < num_state_vec));
    return InnerProd(state[i], state[j]);
}

// ======================================================================

void axpby_c(int i, double scalj, int j, double scalk, int k)
{
  assert((i >= 0) && (i < num_dual_vec));
  assert(j < num_dual_vec);
  assert(k < num_dual_vec);

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
}

void times_vector_c(int i, int j)
{
  assert((i >= 0) && (i < num_dual_vec));
  assert((j >= 0) && (j < num_dual_vec));
  int n;
  for (n = 0; n < num_ceq; n++)
    dual[i](n) *= dual[j](n);
}

void exp_c(int i)
{
  assert((i >= 0) && (i < num_dual_vec));
  int n;
  for (n = 0; n < num_ceq; n++)
    dual[i](n) = exp(dual[i](n));
}

void log_c(int i)
{
  assert((i >= 0) && (i < num_dual_vec));
  int n;
  for (n = 0; n < num_ceq; n++)
    dual[i](n) = log(dual[i](n));
}

void pow_c(int i, double p)
{
  assert((i >= 0) && (i < num_dual_vec));
  int n;
  for (n = 0; n < num_ceq; n++)
    dual[i](n) = pow(dual[i](n), p);
}

double inner_c(int i, int j)
{
  assert((i >= 0) && (i < num_dual_vec));
  assert((j >= 0) && (j < num_dual_vec));
  return InnerProd(dual[i], dual[j]);
}

// ======================================================================
// IDF VECTOR OPERATIONS
// ======================================================================

void zero_targ_state(int at_design)
{
  for (int k = 0; k < 2*nodes; k++)
    design[at_design](num_bspline+k) = 0.0;
}

void zero_real_design(int at_design)
{
  for (int k = 0; k < num_bspline; k++)
    design[at_design](k) = 0.0;
}

void copy_dual_to_targ_state(int take_from, int copy_to)
{
  assert((copy_to >= 0) && (copy_to < num_design_vec));
  assert((take_from >= 0) && (take_from < num_dual_vec));
  for (int k = 0; k < num_bspline; k++)
    design[copy_to](k) = 0.0;
  for (int k = 0; k < 2*nodes; k++)
    design[copy_to](num_bspline+k) = dual[take_from](k);
}

void copy_targ_state_to_dual(int take_from, int copy_to)
{
  assert((take_from >= 0) && (take_from < num_design_vec));
  assert((copy_to >= 0) && (copy_to < num_dual_vec));
  for (int k = 0; k < 2*nodes; k++)
    dual[copy_to](k) = design[take_from](num_bspline+k);
}

// ======================================================================
// OBJECTIVE OPERATIONS
// ======================================================================

boost::python::tuple eval_f(int at_design, int at_state)
{
  assert((at_design >= 0) && (at_design < num_design_vec));
  assert((at_state >= -1) && (at_state < num_state_vec));
  InnerProdVector pts(num_bspline, 0.0);
  GetBsplinePts(at_design, pts);
  nozzle_shape.SetCoeff(pts);
  InnerProdVector area(nodes, 0.0);
  GetCouplingArea(at_design, area);
  cfd_solver.set_area(area);
  int cost = 0; // no precondition calls
  InnerProdVector q(num_dis_var, 0.0);
  GetCFDState(at_state, q);
  cfd_solver.set_q(q);
  return boost::python::make_tuple(
    obj_weight*cfd_solver.CalcInverseDesign(), cost);
}

void eval_dfdx(int at_design, int at_state, int store_here)
{
  int i = at_design;
  int j = at_state;
  int k = store_here;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  assert((k >= 0) && (k < num_design_vec));
  design[k] = 0.0;
}

void eval_dfdw(int at_design, int at_state, int store_here)
{
  int i = at_design;
  int j = at_state;
  int k = store_here;
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
  g_cfd *= obj_weight;
  SetCFDState(k, g_cfd);
  // CSM contribution to objective gradient
  g_cfd = 0.0;
  SetCSMState(k, g_cfd);
}

// ======================================================================
// RESIDUAL OPERATIONS
// ======================================================================

void eval_r(int at_design, int at_state, int store_here)
{
  int i = at_design;
  int j = at_state;
  int k = store_here;
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
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  press *= p_ref;
  press -= press_stag;
  csm_solver.set_press(press);
  csm_solver.set_u(u);
  csm_solver.CalcResidual();
  SetCSMState(k, csm_solver.get_res());
}

void mult_drdx(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
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
      y_coords(nodes, 0.0), u(num_dis_var, 0.0), v_csm(num_dis_var, 0.0);
  GetBsplinePts(i, pts);
  GetCouplingPress(i, press);
  GetCSMState(j, u);
  nozzle_shape.SetCoeff(pts);
  area = nozzle_shape.Area(cfd_solver.get_x_coord());
  CalcYCoords(area, y_coords);
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  press *= p_ref;
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
  //csm_solver.Calc_dydA_Product(u_area, u);
  u_area /= -(2.0*width); // multiply dy/dA *(dA/db * u_pts)
  csm_solver.CalcCmplx_dSdy_Product(u_area, v_cfd); // dS/dy *(dy/db * u_pts)
  v_csm += v_cfd;
  SetCSMState(m, v_csm);
}

void mult_drdw(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
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
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  GetCSMState(k, u_csm);
  csm_solver.Calc_dSdu_Product(u_csm, v_csm);
  SetCSMState(m, v_csm);
}

void mult_drdx_t(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
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
      v_y(nodes, 0.0), v_pts(num_bspline, 0.0);
  GetBsplinePts(i, pts);
  GetCouplingPress(i, press);
  GetCSMState(j, u);
  nozzle_shape.SetCoeff(pts);
  area = nozzle_shape.Area(cfd_solver.get_x_coord());
  CalcYCoords(area, y_coords);
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  press *= p_ref;
  press -= press_stag;
  csm_solver.set_press(press);
  csm_solver.set_u(u);

  // first part: (dS/dp)^T * u_csm
  GetCSMState(k, u);
  csm_solver.CalcTrans_dSdp_Product(u, v_press);
  v_press *= p_ref;
  SetCouplingPress(m, v_press);

  // second part: (dS/d(pts))^T * u_csm
  csm_solver.CalcTransCmplx_dSdy_Product(u, v_area); // (dS/dy)^T * u_csm
  //csm_solver.CalcTrans_dydA_Product(v_area, v_y);
  v_area /= -(2.0*width); // (dy/dA)^T * (dS/dy)^T * u_csm
  v_pts = nozzle_shape.AreaReverseDerivative(
      cfd_solver.get_x_coord(), v_area); // (dA/db)^T * (dy/dA)^T ...
  SetBsplinePts(m, v_pts);
}

void mult_drdw_t(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
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
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  GetCSMState(k, u_csm);
  csm_solver.Calc_dSdu_Product(u_csm, v_csm);
  SetCSMState(m, v_csm);
}

void factor_precond(int at_design, int at_state)
{
  int i = at_design;
  int j = at_state;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  // Evaluate the CFD preconditioner
  InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0);
  GetCouplingArea(i, area);
  GetCFDState(j, q);
  cfd_solver.set_area(area);
  cfd_solver.set_q(q);
  cfd_solver.BuildAndFactorPreconditioner();
}

int apply_precond(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
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
      u_csm(num_dis_var, 0.0), area(nodes, 0.0);
  GetBsplinePts(i, pts);
  nozzle_shape.SetCoeff(pts);
  area = nozzle_shape.Area(cfd_solver.get_x_coord());
  CalcYCoords(area, y_coords);
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  GetCSMState(k, u_csm);
  csm_solver.SolveFor(u_csm, 10000, 1e-3);
  SetCSMState(m, csm_solver.get_u());
  return 1;
}

int apply_precond_t(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
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
      area(nodes, 0.0), u_csm(num_dis_var, 0.0);
  GetBsplinePts(i, pts);
  nozzle_shape.SetCoeff(pts);
  area = nozzle_shape.Area(cfd_solver.get_x_coord());
  CalcYCoords(area, y_coords);
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  GetCSMState(k, u_csm);
  csm_solver.SolveFor(u_csm, 10000, 1e-3);
  SetCSMState(m, csm_solver.get_u());
  return 1;
}

// ======================================================================
// CONSTRAINT OPERATIONS
// ======================================================================

void eval_c(int at_design, int at_state, int store_here)
{
  int i = at_design;
  int j = at_state;
  int k = store_here;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  assert((k >= 0) && (k < num_dual_vec));
  dual[k] = 1.0/kona::kEpsilon;
  // Evaluate the (press - press_coupling) constraint
  InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
      ceq_cfd(nodes, 0.0), press(nodes, 0.0);
  GetCouplingArea(i, area);
  GetCFDState(j, q);
  cfd_solver.set_area(area);
  //cfd_solver.set_q(q);
  cfd_solver.CalcAuxiliaryVariables(q);
  ceq_cfd = cfd_solver.get_press();
  // TEMP: use target pressure for boundaries
  //ceq_cfd(0) = press_targ(0);
  //ceq_cfd(nodes-1) = press_targ(nodes-1);
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
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  csm_solver.set_u(u);
  csm_solver.CalcCoordsAndArea();
  ceq_csm = csm_solver.get_area();
  // TEMP: use area_left and area_right
  //ceq_csm(0) = area_left;
  //ceq_csm(nodes-1) = area_right;
  GetCouplingArea(i, area);
  ceq_csm -= area;
  SetAreaCnstr(k, ceq_csm);
}

void mult_dcdx(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  assert((k >= 0) && (k < num_design_vec));
  assert((m >= 0) && (m < num_dual_vec));
  dual[m] = 1.0/kona::kEpsilon;
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
}

void mult_dcdw(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  assert((k >= 0) && (k < num_state_vec));
  assert((m >= 0) && (m < num_dual_vec));
  dual[m] = 1.0/kona::kEpsilon;
  // component corresponding to (press - press_coupling)
  InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
      u_cfd(num_dis_var, 0.0), v_press(nodes, 0.0);
  GetCouplingArea(i, area);
  GetCFDState(j, q);
  cfd_solver.set_area(area);
  cfd_solver.set_q(q);
  GetCFDState(k, u_cfd);
  cfd_solver.CalcDPressDQProduct(u_cfd, v_press);
  // TEMP: use target pressure
  //v_press(0) = 0.0;
  //v_press(nodes-1) = 0.0;
  SetPressCnstr(m, v_press);
  // component corresponding to (area - area_coupling)
  InnerProdVector u_csm(num_dis_var, 0.0), v_area(nodes, 0.0);
  GetCSMState(k, u_csm);
  csm_solver.Calc_dAdu_Product(u_csm, v_area);
  // TEMP: use area_left and area_right
  //v_area(0) = 0.0;
  //v_area(nodes-1) = 0.0;
  SetAreaCnstr(m, v_area);
}

void mult_dcdx_t(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  assert((k >= 0) && (k < num_dual_vec));
  assert((m >= 0) && (m < num_design_vec));
  design[m] = 1.0/kona::kEpsilon;
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
}

void mult_dcdw_t(int at_design, int at_state, int in_vec, int out_vec)
{
  int i = at_design;
  int j = at_state;
  int k = in_vec;
  int m = out_vec;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  assert((k >= 0) && (k < num_dual_vec));
  assert((m >= 0) && (m < num_state_vec));
  state[m] = 1.0/kona::kEpsilon;
  // component corresponding to (press - press_coupling)
  InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
      u_press(nodes, 0.0), v_cfd(num_dis_var, 0.0);
  GetCouplingArea(i, area);
  GetCFDState(j, q);
  cfd_solver.set_area(area);
  cfd_solver.set_q(q);
  GetPressCnstr(k, u_press);
  // TEMP: use target pressure
  //u_press(0) = 0.0;
  //u_press(nodes-1) = 0.0;
  cfd_solver.CalcDPressDQTransposedProduct(u_press, v_cfd);
  SetCFDState(m, v_cfd);
  // component corresponding to (area - area_coupling)
  InnerProdVector u_area(nodes, 0.0), v_csm(num_dis_var, 0.0);
  GetAreaCnstr(k, u_area);
  // TEMP: use area_left and area_right
  //u_area(0) = 0.0;
  //u_area(nodes-1) = 0.0;
  csm_solver.CalcTrans_dAdu_Product(u_area, v_csm);
  SetCSMState(m, v_csm);
}

// ======================================================================
// SOLVER OPERATIONS
// ======================================================================

int solve_nonlinear(int at_design, int result)
{
  int i = at_design;
  int j = result;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  // Solve the CFD
  InnerProdVector area(nodes, 0.0);
  GetCouplingArea(i, area);
  cfd_solver.set_area(area);
  cfd_solver.InitialCondition(rho_R, rho_u_R, e_R);
  int cost = cfd_solver.NewtonKrylov(20, tol);
  SetCFDState(j, cfd_solver.get_q());
  cfd_solver.WriteTecplot(1.0, 1.0, "cfd_after_solve.dat");
  // Solve the CSM
  InnerProdVector press(nodes, 0.0), pts(num_bspline, 0.0),
      y_coords(nodes, 0.0);
  GetBsplinePts(i, pts);
  GetCouplingPress(i, press);
  nozzle_shape.SetCoeff(pts);
  area = nozzle_shape.Area(cfd_solver.get_x_coord());
  CalcYCoords(area, y_coords);
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  press *= p_ref;
  press -= press_stag;
  csm_solver.set_press(press);
  csm_solver.Solve();
  SetCSMState(j, csm_solver.get_u());
  return cost;
}

int solve_linear(int at_design, int at_state, int rhs, int result, double rel_tol)
{
  int i = at_design;
  int j = at_state;
  int k = rhs;
  int m = result;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  assert((k >= 0) && (k < num_state_vec));
  assert((m >= 0) && (m < num_state_vec));
  // CFD contribution to objective gradient
  InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
      g_cfd(num_dis_var, 0.0), adj_cfd(num_dis_var, 0.0);
  GetCouplingArea(i, area);
  GetCFDState(j, q);
  cfd_solver.set_area(area);
  cfd_solver.set_q(q);
  InnerProdVector pts(num_bspline, 0.0), y_coords(nodes, 0.0),
      u_csm(num_dis_var, 0.0);
  GetCFDState(k, g_cfd);
  int cost = cfd_solver.SolveLinearized(100, adj_tol, g_cfd, adj_cfd);
  SetCFDState(m, adj_cfd);
  // CSM contribution
  // Note: stiffness matrix does not depend on press
  GetBsplinePts(i, pts);
  nozzle_shape.SetCoeff(pts);
  area = nozzle_shape.Area(cfd_solver.get_x_coord());
  CalcYCoords(area, y_coords);
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  GetCSMState(k, g_cfd);
  csm_solver.SolveFor(g_cfd, 10000, adj_tol);
  SetCSMState(m, csm_solver.get_u());
  return cost;
}

int solve_adjoint(int at_design, int at_state, int rhs, int result, double rel_tol)
{
  int i = at_design;
  int j = at_state;
  int k = rhs;
  int m = result;
  assert((i >= 0) && (i < num_design_vec));
  assert((j >= 0) && (j < num_state_vec));
  assert((m >= 0) && (m < num_state_vec));
  assert((k >= 0) && (k < num_state_vec));
  InnerProdVector area(nodes, 0.0), q(num_dis_var, 0.0),
      g_cfd(num_dis_var, 0.0), adj_cfd(num_dis_var, 0.0),
      pts(num_bspline, 0.0), y_coords(nodes, 0.0), u_csm(num_dis_var, 0.0);;
  // CFD contribution to objective gradient
  GetCouplingArea(i, area);
  GetCFDState(j, q);
  cfd_solver.set_area(area);
  cfd_solver.set_q(q);
  GetCFDState(k, g_cfd);
  int cost = cfd_solver.SolveAdjoint(100, adj_tol, g_cfd, adj_cfd);
  SetCFDState(m, adj_cfd);
  // CSM contribution
  // Note: stiffness matrix does not depend on press
  GetBsplinePts(i, pts);
  nozzle_shape.SetCoeff(pts);
  area = nozzle_shape.Area(cfd_solver.get_x_coord());
  CalcYCoords(area, y_coords);
  csm_solver.set_coords(cfd_solver.get_x_coord(), y_coords);
  csm_solver.UpdateMesh();
  GetCSMState(k, g_cfd);
  csm_solver.SolveFor(g_cfd, 10000, adj_tol);
  SetCSMState(m, csm_solver.get_u());
  return cost;
}

// ======================================================================
// BOOST PYTHON MODULE DECLARATION
// ======================================================================

BOOST_PYTHON_MODULE(aerostruct_idf)
{
  using namespace boost::python;

  def("init_mda", init_mda);
  def("alloc_design", alloc_design);
  def("alloc_state", alloc_state);
  def("alloc_dual", alloc_dual);
  def("init_design", init_design);
  def("info_dump", info_dump);

  def("set_design_data", set_design_data);
  def("get_design_data", get_design_data);

  def("set_state_data", set_state_data);
  def("get_state_data", get_state_data);

  def("set_dual_data", set_dual_data);
  def("get_dual_data", get_dual_data);

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

  def("axpby_c", axpby_c);
  def("times_vector_c", times_vector_c);
  def("exp_c", exp_c);
  def("log_c", log_c);
  def("pow_c", pow_c);
  def("inner_c", inner_c);

  def("zero_targ_state", zero_targ_state);
  def("zero_real_design", zero_real_design);
  def("copy_dual_to_targ_state", copy_dual_to_targ_state);
  def("copy_targ_state_to_dual", copy_targ_state_to_dual);

  def("eval_f", eval_f);
  def("eval_dfdx", eval_dfdx);
  def("eval_dfdw", eval_dfdw);

  def("eval_r", eval_r);
  def("mult_drdx", mult_drdx);
  def("mult_drdx_t", mult_drdx_t);
  def("mult_drdw", mult_drdw);
  def("mult_drdw_t", mult_drdw_t);

  def("eval_c", eval_c);
  def("mult_dcdx", mult_dcdx);
  def("mult_dcdx_t", mult_dcdx_t);
  def("mult_dcdw", mult_dcdw);
  def("mult_dcdw_t", mult_dcdw_t);

  def("factor_precond", factor_precond);
  def("apply_precond", apply_precond);
  def("apply_precond_t", apply_precond_t);

  def("solve_nonlinear", solve_nonlinear);
  def("solve_linear", solve_linear);
  def("solve_adjoint", solve_adjoint);
};
