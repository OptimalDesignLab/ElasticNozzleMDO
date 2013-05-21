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

  int nnp = 41; //81;
	int order = 3;
	AeroStructMDA asmda(nnp, order);

	printf("Initializing problem...\n");
	asmda.InitializeTestProb();

#if 0
	printf("Validating MDA product...\n");
	asmda.TestMDAProduct();
#else
	printf("Starting solver...\n");
	int precond_calls = asmda.NewtonKrylov(10, 1.e-7);

        cout << "Number of MDA preconditioner calls = " << precond_calls << endl;
        
	printf("Generating tecplot...\n");
	double rho_ref = 1.14091202011454;
	double p = 9.753431315656936E4;
	double a_ref = sqrt(kGamma*p/rho_ref);
	asmda.GetTecplot(rho_ref, a_ref);
#endif
#if 0
	printf("Validating MDA product...\n");
	asmda.TestMDAProduct();
#endif
        
	return 0;
}
