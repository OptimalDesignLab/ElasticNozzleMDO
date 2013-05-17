/**
 * \file MDA_test.cpp
 * \brief Aero-Structural MDA test executable
 * \author  Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

#include <stdio.h>
#include "./aerostruct.hpp"
using namespace std;

// =====================================================================

int main() {

	int nnp = 50;
	int order = 4;
	AeroStructMDA asmda(nnp, order);

	asmda.InitializeTestProb();

	asmda.NewtonKrylov(100, 1.e-10);

	double rho_ref = 1.14091202011454;
	double a_ref = sqrt(kGamma*p/rho);
	asmda.GetTecplot(rho_ref, a_ref);

	return 0;
}