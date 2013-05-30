/**
 * \file constants.hpp
 * \brief parameters and constants for nozzle MDO problem
 * \author Jason Hicken <jason.hicken@gmail.com>, Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

#pragma once

// nozzle parameters
const double length = 1.0;
const double height = 2.0;
const double width = 1.0;
const double x_min = 0.0;
const double x_max = x_min + length;
const double area_left = 2.0;
const double area_right = 1.9; //1.75;
const double area_mid = 1.8; //1.5;
const double y_left = 0.5*(height - area_left/width);
const double y_right = 0.5*(height - area_right/width);

// discretization parameters
const int nodes = 21;

// CFD discretization parameters
const int order = 3;
const bool sbp_quad = true; // use SBP norm for quadrature

// CFD solution parameters
const double kAreaStar = 0.8;
const bool subsonic = true;
const double kTempStag = 300.0;
const double kPressStag = 100000;
const double kRGas = 287.0;

// CSM material parameters
const double E = 1000000000.0;   // Young's modulus
const double thick = 0.01;        // fixed beam element thickness

// tolerances for primal and adjoint problems
const double tol = 1.e-12;
const double adj_tol = 1.e-10;
