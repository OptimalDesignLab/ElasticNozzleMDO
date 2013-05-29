/**
 * \file MDA_test.cpp
 * \brief Aero-Structural MDA test executable
 * \author  Alp Dener <alp.dener@gmail.com>
 * \version 1.0
 */

#include <stdio.h>
#include "../aerostruct.hpp"
using namespace std;

// =====================================================================

int main() {

  int nnp = 21; //41; //81;
	int order = 3;
	AeroStructMDA asmda(nnp, order);

	printf("Initializing problem...\n");
	asmda.InitializeTestProb();

#if 0
	printf("Validating MDA product...\n");
	asmda.TestMDAProduct();
#else
	printf("Starting solver...\n");
	int precond_calls = asmda.NewtonKrylov(30, 1.e-8);

        cout << "Number of MDA preconditioner calls = " << precond_calls << endl;

        //asmda.PrintDisplacements();
        
	printf("Generating tecplot...\n");
	double rho_ref = 1.14091202011454;
	double p = 9.753431315656936E4;
	double a_ref = sqrt(kGamma*p/rho_ref);
        rho_ref = 1.0;
        a_ref = 1.0;
	asmda.GetTecplot(rho_ref, a_ref);

        //asmda.GridTest();
        
#endif
        
#if 0
	printf("Validating MDA product...\n");
	asmda.TestMDAProduct();
        asmda.TestMDATransposedProduct();
#endif

#if 1
        printf("Testing MDA adjoint solver...\n");
        InnerProdVector dJdu(6*nnp, 0.0), psi(6*nnp, 0.0);
        for (int i = 0; i < 3*nnp; i++)
          dJdu(i) = 1.0;
        asmda.SolveAdjoint(1000, 1e-8, dJdu, psi);

        // for visualization
        asmda.set_u(psi);        
#endif
            
	return 0;
}
