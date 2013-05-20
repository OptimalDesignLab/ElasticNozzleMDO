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

  int nnp = 21; //81;
	int order = 4;
	AeroStructMDA asmda(nnp, order);

	printf("Initializing problem...\n");
	asmda.InitializeTestProb();

#if 1
	printf("Validating MDA product...\n");
	asmda.TestMDAProduct();
#else
	printf("Starting solver...\n");
	asmda.NewtonKrylov(100, 1.e-10);

	printf("Generating tecplot...\n");
	double rho_ref = 1.14091202011454;
	double p = 9.753431315656936E4;
	double a_ref = sqrt(kGamma*p/rho_ref);
	asmda.GetTecplot(rho_ref, a_ref);
#endif

	return 0;
}
