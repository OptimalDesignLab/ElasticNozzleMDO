/**
 * \file constants.hpp
 * \brief parameters and constants for nozzle MDO problem
 * \author Jason Hicken <jason.hicken@gmail.com>, Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

// nozzle parameters
static const double length = 1.0;
static const double height = 2.0;
static const double width = 1.0;
static const double x_min = 0.0;
static const double x_max = x_min + length;
static const double area_left = 2.0;
static const double area_right = 1.75;
static const double area_mid = 1.5;
static const double y_left = 0.5*(height - area_left/width);
static const double y_right = 0.5*(height - area_right/width);

// discretization parameters
static const int nodes = 21;

// CFD discretization parameters
static const int order = 3;
static const bool sbp_quad = true; // use SBP norm for quadrature

// CFD solution parameters
static const double kAreaStar = 0.8;
static const bool subsonic = true;
static const double kTempStag = 300.0;
static const double kPressStag = 100000;
static const double kRGas = 287.0;
static double rho_R, rho_u_R, e_R;
static double p_ref;

// CSM material parameters
double E = 100000000.0;   // Young's modulus
double t = 0.01;        // fixed beam element thickness
