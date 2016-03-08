/**
 * \file krylov.cpp
 * \brief definitions of templated Krylov-subspace methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 * \version 1.0
 */

#include <string>
#include <vector>
#include <boost/numeric/ublas/vector_proxy.hpp>
//#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "./krylov.hpp"

namespace kona {

/* \brief LAPACK function declarations
 */
extern "C" void dsyev_(char * JOBZ, char * UPLO, int * N, double * A, int * LDA,
                       double * W, double * WORK, int * LWORK, int * INFO);

extern "C" void dgeqrf_(int * M, int * N, double * A, int * LDA, double * tau,
                        double * WORK, int * LWORK, int * INFO);

extern "C" void dtrtrs_(char * UPLO, char * TRANS, char * DIAG, int * N,
                        int * NRHS, double * A, int * LDA, double * B, int * LDB,
                        int * INFO);

extern "C" void dormqr_(char * SIDE, char * TRANS, int * M, int * N, int * K,
                        double * A, int * LDA, double * TAU, double * C,
                        int * LDC, double * WORK, int * LWORK, int * INFO);

extern "C" void dgesvd_(char * JOBU, char * JOBVT, int * M, int * N, double * A,
                        int * LDA, double * S, double * U, int * LDU,
                        double * VT, int * LDVT, double * WORK, int * LWORK,
                        int * INFO);

extern "C" void dpotrf_(char * UPLO, int * N, double * A, int * LDA, int * INFO);

extern "C" void dtrsm_(char * SIDE, char * UPLO, char * TRANS, char * DIAG,
                       int * N, int * M, double * ALPHA, double * A, int * LDA,
                       double * B, int * LDB);

extern "C" void dgesv_(int * N, int * NRHS, double * A, int * LDA, int * IPIV,
                       double * B, int * LDB, int * INFO);

extern "C" void dgelsd_(int * M, int * N, int * NRHS, double * A, int * LDA,
                        double * B, int * LDB, double * S, double * RCOND,
                        int * RANK, double * WORK, int * LWORK, int * IWORK,
                        int * INFO);

extern "C" void dgels_(char * TRANS, int * M, int * N, int * NRHS, double * A,
                       int * LDA, double * B, int * LDB, double * WORK,
                       int * LWORK, int * INFO);


// ======================================================================

double sign(const double & x, const double & y) {
  if (y == 0.0) {
    return 0.0;
  } else {
    return (y < 0 ? -fabs(x) : fabs(x));
  }
}

// ======================================================================

double CalcEpsilon(const double & eval_at_norm,
                   const double & mult_by_norm) {

  if ( (mult_by_norm < kEpsilon*eval_at_norm) ||
       (mult_by_norm < kEpsilon) ) {
    // multiplying vector is zero or essentially zero
    return 1.0; //0.01; //sqrt(kEpsilon);
  } else if (eval_at_norm < kEpsilon*mult_by_norm) {
    // multiplying vector dominates, so treat eval_at vector like zero
    return sqrt(kEpsilon)/mult_by_norm;
  } else {
    return sqrt(kEpsilon)*eval_at_norm/mult_by_norm;
  }


#if 0
  if ( (mult_by_norm < kEpsilon*eval_at_norm) ||
       ( (mult_by_norm < kEpsilon) && (eval_at_norm < kEpsilon) ) ) {
    // multiplying vector is zero, or is essentially zero
    return 1.0; //sqrt(kEpsilon);
  } else if (eval_at_norm < kEpsilon*mult_by_norm) {
    // multiplying vector dominates, so treat eval_at vector like zero
    return sqrt(kEpsilon)/mult_by_norm;
  } else {
    return sqrt(kEpsilon)*eval_at_norm/mult_by_norm;
  } 
#endif
}

// ======================================================================

void eigenvalues(const int & n, const ublas::matrix<double> & A,
                 ublas::vector<double> & eig) {
  if (n < 1) {
    cerr << "krylov.cpp (eigenvalues): "
	 << "matrix dimension must be greater than 0." << endl;
    throw(-1);
  }
  if ( (A.size1() < n) || (A.size2() < n) ) {
    cerr << "krylov.cpp (eigenvalues): "
         << "given matrix has fewer rows/columns than given dimension." << endl;
    throw(-1);
  }
  
  // Asym stores the symmetric part of A in column-major ordering (actually,
  // ordering doesn't matter for symmetric matrix)
  ublas::vector<double> Asym(n*n);
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {      
      Asym(i*n + j) = 0.5*(A(i,j) + A(j,i));
      Asym(j*n + i) = Asym[i*n + j];
    }
  }
  
  char jobz = 'N';
  char uplo = 'U';
  int Adim = n; // need copy, because n is const int &
  int lwork = 3*n;
  ublas::vector<double> work(lwork);
  int info;
  dsyev_(&jobz, &uplo, &Adim, &*Asym.begin(), &Adim, &*eig.begin(),
         &*work.begin(), &lwork, &info);
  if (info != 0) {
    cerr << "krylov.cpp (eigenvalues): "
	 << "LAPACK routine dsyev failed with info = " << info << endl;
    throw(-1);
  }
}

// ======================================================================

void eigenvaluesAndVectors(const int & n, const ublas::matrix<double> & A,
                           ublas::vector<double> & eig,
                           ublas::matrix<double> & E) {
  if (n < 1) {
    cerr << "krylov.cpp (eigenvaluesAndVectors): "
	 << "matrix dimension must be greater than 0." << endl;
    throw(-1);
  }
  if ( (A.size1() < n) || (A.size2() < n) ) {
    cerr << "krylov.cpp (eigenvaluesAndVectors): "
         << "given matrix has fewer rows/columns than given dimension." << endl;
    throw(-1);
  }
  if ( (E.size1() < n) || (E.size2() < n) ) {
    cerr << "krylov.cpp (eigenvaluesAndVectors): "
         << "Eigenvector matrix has fewer rows/columns than given dimension."
         << endl;
    throw(-1);
  }
  
  // Asym stores the symmetric part of A in column-major ordering (actually,
  // ordering doesn't matter for symmetric matrix)
  ublas::vector<double> Asym(n*n);
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {      
      Asym(i*n + j) = 0.5*(A(i,j) + A(j,i));
      Asym(j*n + i) = Asym[i*n + j];
    }
  }
  
  char jobz = 'V';
  char uplo = 'U';
  int Adim = n; // need copy, because n is const int &
  int lwork = 3*n;
  ublas::vector<double> work(lwork);
  int info;
  dsyev_(&jobz, &uplo, &Adim, &*Asym.begin(), &Adim, &*eig.begin(),
         &*work.begin(), &lwork, &info);
  if (info != 0) {
    cerr << "krylov.cpp (eigenvalues): "
	 << "LAPACK routine dsyev failed with info = " << info << endl;
    throw(-1);
  }
  for (int j = 0; j < n; j++) 
    for (int i = 0; i < n; i++)
      E(i,j) = Asym(j*n + i);
}

// ======================================================================

void factorQR(const int & nrow, const int & ncol,
              const ublas::matrix<double> & A, ublas::vector<double> & QR) {
  if ( (nrow < 1) || (ncol < 1) ) {
    cerr << "krylov.cpp (factorQR): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if ( (A.size1() < nrow) || (A.size2() < ncol) ) {
    cerr << "krylov.cpp (factorQR): "
         << "given matrix has fewer rows/columns than given dimensions." << endl;
    throw(-1);
  }
  if (nrow < ncol) {
    cerr << "krylov.cpp (factorQR): "
         << "number of rows must be greater than or equal the number of columns."
         << endl;
    throw(-1);
  }

  // Copy A into QR in column-major ordering
  QR.resize(nrow*ncol + ncol);
  for (int j = 0; j < ncol; j++)
    for (int i = 0; i < nrow; i++)
      QR(j*nrow + i) = A(i,j);
  
  int m = nrow;
  int n = ncol;
  int lwork = ncol;
  ublas::vector<double> work(lwork);
  int info;
  ublas::vector<double>::iterator tau = QR.end() - ncol;
  dgeqrf_(&m, &n, &*QR.begin(), &m, &*tau, &*work.begin(), &lwork, &info);
  if (info != 0) {
    cerr << "krylov.cpp (factorQR): "
	 << "LAPACK routine dgeqrf failed with info = " << info << endl;
    throw(-1);
  }
}

// ======================================================================

void solveR(const int & nrow, const int & ncol, ublas::vector<double> & QR,
            const ublas::vector<double> & b, ublas::vector<double> & x,
            const bool & transpose) {
  if ( (nrow < 1) || (ncol < 1) ) {
    cerr << "krylov.cpp (invertR): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if (nrow < ncol) {
    cerr << "krylov.cpp (invertR): "
         << "number of rows must be greater than or equal the number of columns."
         << endl;
    throw(-1);
  }
  if ( (b.size() < ncol) || (x.size() < ncol) ) {
    cerr << "krylov.cpp (invertR): "
         << "vector b and/or x are smaller than size of R." << endl;
    throw(-1);
  }

  // copy rhs vector b into x (which is later overwritten with solution)
  for (int i = 0; i < ncol; i++)
    x(i) = b(i);
  
  char uplo = 'U';
  char trans = 'N';
  if (transpose) trans = 'T';
  char diag = 'N';
  int m = nrow;
  int n = ncol;
  int nrhs = 1;
  int info;
  dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, &*QR.begin(), &m, &*x.begin(), &n,
          &info);
  if (info > 0) {
    cerr << "krylov.cpp (solveR): "
         << "the " << info << "-th diagonal entry of R is zero" << endl;
    throw(-1);
  } else if (info < 0) {
    cerr << "krylov.cpp (solveR): "
	 << "LAPACK routine dtrtrs failed with info = " << info << endl;
    throw(-1);
  }
}

// ======================================================================

void applyQ(const int & nrow, const int & ncol, ublas::vector<double> & QR,
            const ublas::vector<double> & b, ublas::vector<double> & x,
            const bool & transpose) {
  if ( (nrow < 1) || (ncol < 1) ) {
    cerr << "krylov.cpp (applyQ): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if (nrow < ncol) {
    cerr << "krylov.cpp (applyQ): "
         << "number of rows must be greater than or equal the number of columns."
         << endl;
    throw(-1);
  }
  if ( (b.size() < nrow) || (x.size() < nrow) ) {
    cerr << "krylov.cpp (applyQ): "
         << "vector b and/or x are smaller than nrow." << endl;
    throw(-1);
  }
  
  // copy rhs vector b into x (which is later overwritten with solution)
  for (int i = 0; i < nrow; i++)
    x(i) = b(i);
  
  char side = 'L';
  char trans = 'N';
  if (transpose) trans = 'T';
  int m = nrow;
  int n = ncol;
  int nrhs = 1;
  int lwork = nrow;
  ublas::vector<double> work(lwork);
  ublas::vector<double>::iterator tau = QR.end() - ncol;
  int info;
  dormqr_(&side, &trans, &m, &nrhs, &n, &*QR.begin(), &m, &*tau, &*x.begin(),
          &m, &*work.begin(), &lwork, &info);
  if (info != 0) {
    cerr << "krylov.cpp (applyQ): "
	 << "LAPACK routine dormqr failed with info = " << info << endl;
    throw(-1);
  }

}

// ======================================================================

void factorCholesky(const int & n, const ublas::matrix<double> & A,
                    ublas::vector<double> & UTU) {
  if (n < 1) {
    cerr << "krylov.cpp (factorCholesky): "
	 << "matrix dimension must be greater than 0." << endl;
    throw(-1);
  }
  if ( (A.size1() < n) || (A.size2() < n) ) {
    cerr << "krylov.cpp (factorCholesky): "
         << "given matrix has fewer rows/columns than given dimension." << endl;
    throw(-1);
  }
  
  // UTU stores the symmetric part of A in column-major ordering (actually,
  // ordering doesn't matter for symmetric matrix)
  UTU.resize(n*n);
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {      
      UTU(i*n + j) = 0.5*(A(i,j) + A(j,i));
      UTU(j*n + i) = UTU(i*n + j);
    }
  }
  
  char uplo = 'U';
  int Adim = n; // need copy, because n is const int &
  int info;
  dpotrf_(&uplo, &Adim, &*UTU.begin(), &Adim, &info);
  if (info > 0) {
    throw true;
  } else if (info < 0) {
    cerr << "krylov.cpp (factorCholesky): "
	 << "LAPACK routine dpotrf failed with info = " << info << endl;
    throw(-1);
  }
}

// ======================================================================

void solveU(const int & n, ublas::vector<double> & UTU,
            const ublas::vector<double> & b, ublas::vector<double> & x,
            const bool & transpose) {
  if (n < 1) {
    cerr << "krylov.cpp (applyU): "
	 << "matrix dimension must be greater than 0." << endl;
    throw(-1);
  }
  if ( (b.size() < n) || (x.size() < n) ) {
    cerr << "krylov.cpp (applyU): "
         << "vector b and/or x are smaller than n." << endl;
    throw(-1);
  }

  // copy rhs vector b into x (which is later overwritten with solution)
  for (int i = 0; i < n; i++)
    x(i) = b(i);
  char side = 'L';
  char uplo = 'U';
  char trans = 'N';
  if (transpose) trans = 'T';
  char diag = 'N';
  int nrow = n;
  int nrhs = 1;
  double one = 1.0;
  dtrsm_(&side, &uplo, &trans, &diag, &nrow, &nrhs, &one, &*UTU.begin(),
         &nrow, &*x.begin(), &nrow);
}

// ======================================================================

void computeSVD(const int & nrow, const int & ncol,
                const ublas::matrix<double> & A,
                ublas::vector<double> & Sigma, ublas::vector<double> & U, 
                ublas::vector<double> & VT, const bool& All_of_U) {
  if ( (nrow < 1) || (ncol < 1) ) {
    cerr << "krylov.cpp (computeSVD): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if ( (A.size1() < nrow) || (A.size2() < ncol) ) {
    cerr << "krylov.cpp (computeSVD): "
         << "given matrix has fewer rows/columns than given dimensions." << endl;
    throw(-1);
  }
  if (nrow < ncol) {
    cerr << "krylov.cpp (computeSVD): "
         << "number of rows must be greater than or equal the number of columns."
         << endl;
    throw(-1);
  }

  int info;
  if (All_of_U) {
    // Resize output vectors and copy A into work array
    Sigma.resize(ncol);
    U.resize(nrow*nrow);
    VT.resize(ncol*ncol);
    ublas::vector<double> Acpy(nrow*ncol);
    for (int j = 0; j < ncol; j++)
      for (int i = 0; i < nrow; i++)
        Acpy(j*nrow + i) = A(i,j);    
    char jobu = 'A';
    char jobvt = 'A';
    int m = nrow;
    int n = ncol;
    int lwork = std::max(3*ncol + nrow, 5*ncol);
    ublas::vector<double> work(lwork);
    dgesvd_(&jobu, &jobvt, &m, &n, &*Acpy.begin(), &m, &*Sigma.begin(),
            &*U.begin(), &m, &*VT.begin(), &n, &*work.begin(), &lwork, &info);
  } else {
    // Resize output vectors, and copy A into U in column-major ordering
    Sigma.resize(ncol);
    U.resize(nrow*ncol);
    VT.resize(ncol*ncol);  
    for (int j = 0; j < ncol; j++)
      for (int i = 0; i < nrow; i++)
        U(j*nrow + i) = A(i,j);
    
    char jobu = 'O';
    char jobvt = 'A';
    int m = nrow;
    int n = ncol;
    double * Ujunk;
    int ldu = 1;
    int lwork = std::max(3*ncol + nrow, 5*ncol);
    ublas::vector<double> work(lwork);
    dgesvd_(&jobu, &jobvt, &m, &n, &*U.begin(), &m, &*Sigma.begin(), Ujunk, &ldu,
            &*VT.begin(), &n, &*work.begin(), &lwork, &info);
  }
  if (info != 0) {
    cerr << "krylov.cpp (computeSVD): "
         << "LAPACK routine dgesvd failed with info = " << info << endl;
    throw(-1);
  }
}
  
// ======================================================================

void applyGivens(const double & s, const double & c,
		 double & h1, double & h2) {
  double temp = c*h1 + s*h2;
  h2 = c*h2 - s*h1;
  h1 = temp;
}

// ======================================================================

void generateGivens(double & dx, double & dy, double & s, double & c) {
  if ( (dx == 0.0) && (dy == 0.0) ) {
    c = 1.0;
    s = 0.0;

  } else if ( fabs(dy) > fabs(dx) ) {
    double tmp = dx/dy;
    dx = sqrt(1.0 + tmp*tmp);
    s = sign(1.0/dx, dy);
    c = tmp*s;

  } else if ( fabs(dy) <= fabs(dx) ) {
    double tmp = dy/dx;
    dy = sqrt(1.0 + tmp*tmp);
    c = sign(1.0/dy, dx);
    s = tmp*c;

  } else {
    // dx and/or dy must be invalid
    dx = 0.0;
    dy = 0.0;
    c = 1.0;
    s = 0.0;
  }
  dx = fabs(dx*dy);
  dy = 0.0;
}

// ======================================================================

void solveReduced(const int & n, const ublas::matrix<double> & A,
                  const ublas::vector<double> & rhs, ublas::vector<double> & x) {
  if (n < 1) {
    cerr << "krylov.cpp (solveReduced): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if ( (A.size1() < n) || (A.size2() < n) ) {
    cerr << "krylov.cpp (solveReduced): "
         << "given matrix has fewer rows/columns than given dimensions." << endl;
    throw(-1);
  }
  if (rhs.size() < n) {
    cerr << "krylov.cpp (solveReduced): "
         << "given rhs has fewer rows than given dimensions." << endl;
    throw(-1);
  }
  if (x.size() < n) {
    cerr << "krylov.cpp (solveReduced): "
         << "given x has fewer rows than given dimensions." << endl;
    throw(-1);
  }

  // LU stores A in column-major ordering (eventually, LU will hold the
  // LU-factorization of A)
  ublas::vector<double> LU(n*n, 0.0);
  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++)
      LU(j*n + i) = A(i,j);

  // Y stores RHS in column-major ordering (Y is overwritten with solution)
  ublas::vector<double> Y(n, 0.0);
  for (int i = 0; i < n; i++)
    Y(i) = rhs(i);

  int Arow = n; // need copy, because n is const int &
  int RHScol = 1; // similarly
  int info;
  ublas::vector<int> ipiv(n, 0.0);
  dgesv_(&Arow, &RHScol, &*LU.begin(), &Arow, &*ipiv.begin(), &*Y.begin(), &Arow,
         &info);
  if (info != 0) {
    cerr << "krylov.cpp (solveReduced): "
	 << "LAPACK routine dgesv failed with info = " << info << endl;
    throw(-1);
  }
  // put solution into x
  for (int i = 0; i < n; i++)
    x(i) = Y(i);
}

// ======================================================================

void solveReducedMultipleRHS(const int & n,
                             const ublas::matrix<double> & A,
                             const int & nrhs,
                             const ublas::matrix<double> & RHS,
                             ublas::matrix<double> & X) {

  if ( (n < 1) || (nrhs < 1) ) {
    cerr << "krylov.cpp (solveReducedMultipleRHS): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if ( (A.size1() < n) || (A.size2() < n) ) {
    cerr << "krylov.cpp (solveReducedMultipleRHS): "
         << "given matrix has fewer rows/columns than given dimensions." << endl;
    throw(-1);
  }
  if ( (RHS.size1() < n) || (RHS.size2() < nrhs) ) {
    cerr << "krylov.cpp (solveReducedMultipleRHS): "
         << "given RHS has fewer rows/columns than given dimensions." << endl;
    throw(-1);
  }
  if ( (X.size1() < n) || (X.size2() < nrhs) ) {
    cerr << "krylov.cpp (solveReducedMultipleRHS): "
         << "given X has fewer rows/columns than given dimensions." << endl;
    throw(-1);
  }

  // LU stores A in column-major ordering (eventually, LU will hold the
  // LU-factorization of A)
  ublas::vector<double> LU(n*n, 0.0);
  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++)
      LU(j*n + i) = A(i,j);

  // Y stores RHS in column-major ordering (Y is overwritten with solution)
  ublas::vector<double> Y(n*nrhs, 0.0);
  for (int j = 0; j < nrhs; j++)
    for (int i = 0; i < n; i++)
      Y(j*n + i) = RHS(i,j);

  int Arow = n; // need copy, because n is const int &
  int RHScol = nrhs; // similarly
  int info;
  double rcond = 1.e-12;
  int rank;
  ublas::vector<double> S(n, 0.0);
  ublas::vector<int> iwork(20*n, 0);

  int lwork = -1;
  ublas::vector<double> work(1, 0.0);
  dgelsd_(&Arow, &Arow, &RHScol, &*LU.begin(), &Arow, &*Y.begin(), &Arow,
          &*S.begin(), &rcond, &rank, &*work.begin(), &lwork, &*iwork.begin(),
          &info);
  lwork = static_cast<int>(work(0));
  work.resize(lwork);  
  dgelsd_(&Arow, &Arow, &RHScol, &*LU.begin(), &Arow, &*Y.begin(), &Arow,
          &*S.begin(), &rcond, &rank, &*work.begin(), &lwork, &*iwork.begin(),
          &info);
  if (info != 0) {
    cerr << "krylov.cpp (solveReducedMultipleRHS): "
	 << "LAPACK routine dgelsd failed with info = " << info << endl;
    throw(-1);
  }
  
#if 0
  int Arow = n; // need copy, because n is const int &
  int RHScol = nrhs; // similarly
  int info;
  ublas::vector<int> ipiv(n, 0.0);
  dgesv_(&Arow, &RHScol, &*LU.begin(), &Arow, &*ipiv.begin(), &*Y.begin(), &Arow,
         &info);
  if (info != 0) {
    cerr << "krylov.cpp (solveReducedMultipleRHS): "
	 << "LAPACK routine dgesv failed with info = " << info << endl;
    throw(-1);
  }
#endif
  
  // put solution into X
  for (int j = 0; j < nrhs; j++)
    for (int i = 0; i < n; i++)
      X(i,j) = Y(j*n + i);
}

// ======================================================================

void solveReducedHessenberg(const int & n,
                            const ublas::matrix<double> & Hsbg,
                            const ublas::vector<double> & rhs,
                            ublas::vector<double> & x) {
  // initialize...
  x = rhs;
  // ... and backsolve
  for (int i = n-1; i >= 0; i--) {
    x(i) /= Hsbg(i,i);
    for (int j = i-1; j >= 0; j--) {
      x(j) -= Hsbg(j,i)*x(i);
    }
  }
}

// ======================================================================
#if 0
void solveTrustReduced(const int & n, const ublas::matrix<double> & H,
                       const ublas::matrix<double> & B, const double & mineig,
                       const double & radius, const ublas::vector<double> & g,
                       ublas::vector<double> & y, double & lambda,
                       double & res) {
  if ( n < 0 ) {
    cerr << "krylov.cpp (solveTrustReduced): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if (radius < 0.0) {
    cerr << "krylov.cpp (solveTrustReduced): "
         << " trust-region radius must be nonnegative: radius = " << radius
         << endl;
    throw(-1);
  }
  if ( (H.size1() < n+1) || (H.size2() < n) ||
       (B.size1() < n+1) || (B.size2() < n) ) {
    cerr << "krylov.cpp (solveTrustReduced): "
         << "H and/or B matrix are smaller than necessary." << endl;
    throw(-1);
  }
  if ( (g.size() < 2*n+1) || (y.size() < n) ) {
    cerr << "krylov.cpp (solveTrustReduced): "
         << "g and/or y vectors are smaller than necessary." << endl;
    throw(-1);
  }

  if (mineig >= 0.0) {
    // Hessian is semi-definite on span(Z), so solve for y and check if ||y|| is
    // in trust region radius
    ublas::vector<double> QR;
    factorQR(n+1, n, H, QR);
    ublas::vector<double> rhs(n+1);
    applyQ(n+1, n, QR, g, rhs, true);
    solveR(n+1, n, QR, rhs, y);
    double norm_y = 0.0;
    for (int i = 0; i < n; i++)
      norm_y += y(i)*y(i);
    norm_y = sqrt(norm_y);
    if (norm_y < radius) {
      res = fabs(rhs(n));
      lambda = 0.0;
      return;
    }
    cout << "\t\tsolveTrustReduced: norm_2(y) = " << norm_y << " > " << radius
         << endl;    
  }
  // if we get here, either the Hessian is semi-definite or ||y|| > radius

  cout << "n = " << n << endl;
  if (n == 1) {
    y(1) = radius;
    lambda = (-g(0)*B(0,0) - mineig)/(B(0,0)*B(0,0) + B(1,0)*B(1,0));
    cout << "\t\tsolveTrustReduced (n=1): lambda = " << lambda << endl;
    //double r0 = g(0) - H(0,0) - B(0,0)*lambda;
    //double r1 = -H(1,0) - B(1,0)*lambda;
    double r0 = g(0) + H(0,0) + B(0,0)*lambda;
    double r1 = H(1,0) + B(1,0)*lambda;
    res = sqrt(r0*r0 + r1*r1);
    return;
  }
  
  // find SVD of B, and find index s such that
  //     1 = Sigma(0) = --- = Sigma(s-1) > Sigma(s)  
  // Note: for the given B, the Sigma should lie between 0 and 1 (see pg 603,
  // Golub and van Loan); check this here too
  ublas::vector<double> Sigma, P, UT;
  computeSVD(n+1, n, B, Sigma, P, UT);
  for (int s = 0; s < n; s++) {
    if ( (Sigma(s) < - 1E+4*kEpsilon) || (Sigma(s) > 1.0+1E+4*kEpsilon) ) {
      cerr << "krylov.cpp (solveTrustReduced): "
           << "Singular values of B are not in [0,1]. "
           << "check that Z and V have orthogonal columns" << endl;
      cout << "B sigma = ";
      for (int i = 0; i < n; i++)
        cout << Sigma[i] << " ";
      cout << endl;
      throw(-1);
    }
  }

  //cout << "\t\tsolveTrustReduced: after compute SVD..." << endl;
  
  int s;
  cout << "\t\tsolveTrustReduced: singular values = ";
  for (s = 0; s < n; s++) {
    if (fabs(Sigma(s) - 1.0) > 1E+4*kEpsilon) break;
    cout << Sigma(s) << " ";
  }
  cout << endl;

  cout << "\t\tsolveTrustReduced: n, s, n-s = " << n << ", " << s << ", "
       << n-s << endl;
  
  // compute the matrix C = W^T * Z = N*UT(s:n-1)*(I - B^T*B)
  // where N^{-1} = diag(sqrt(1 - Sigma(s:n-1)))
  ublas::matrix<double> C;
  if (s < n) {
    C.resize(n-s,n);
    // Step 1: compute P := I - B^T*B (stored column major)
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        P(i+j*n) = 0.0;
        for (int k = 0; k < n+1; k++)
          P(i+j*n) += B(k,i)*B(k,j);
      }
      P(j+j*n) += 1.0;
    }
    // Step 2: compute C := QT(s:n-1)*P
    for (int i = 0; i < n-s; i++) {
      for (int j = 0; j < n; j++) {
        C(i,j) = 0.0;
        for (int k = 0; k < n; k++)
          C(i,j) += UT(s+i+k*n)*P(k+j*n);
      }
    }
    // Step 3: compute C := N*C
    for (int i = 0; i < n-s; i++) {
      double fac = 1.0/sqrt(1.0 - Sigma(s+i)*Sigma(s+i));
      for (int j = 0; j < n; j++)
        C(i,j) *= fac;
    }
  }

  //cout << "solveTrustReduced: after computing C..." << endl;
  
  // initialize Lagrange multiplier: if first time through this function, use
  // minimum eigenvalue to estimate (or zero if eigenvalue positive).  If this
  // is not the first time, use previous solution
#if 0
  if (lambda < kEpsilon) 
    lambda = std::max(-2.0*mineig, 0.0); // first time
#endif
  //lambda = std::max(-2.0*mineig, 0.0);
  lambda = std::max(-2.0*mineig, 2.0*mineig);
  
  cout << "H = " << endl;
  cout << H(0,0) << ", " << H(0,1) << endl;
  cout << H(1,0) << ", " << H(1,1) << endl;
  cout << H(2,0) << ", " << H(2,1) << endl;
  cout << "B = " << endl;
  cout << B(0,0) << ", " << B(0,1) << endl;
  cout << B(1,0) << ", " << B(1,1) << endl;
  cout << B(2,0) << ", " << B(2,1) << endl;
  cout << "g = " << endl;
  cout << g(0) << endl;
  cout << g(1) << endl;
  cout << g(2) << endl;


    
  
  // Apply Newton's method to find lambda
  int maxNewt = 15;
  double tol = 1.e-10;
  ublas::matrix<double> Hhat(2*n+1-s,n);

#if 0
  // TEMP: data to plot relationship between lambda and |y|
  int m = 200;
  cout << "lambda = [";
  for (int i = 0; i < m; i++) {
    cout << 2.0*mineig -4.0*mineig*static_cast<double>(i)/static_cast<double>(m-1) << " ";
  }
  cout << "]" << endl;
  cout << "norm_y = [";
  for (int i = 0; i < m; i++) {
    lambda = 2.0*mineig -4.0*mineig*static_cast<double>(i)/static_cast<double>(m-1);
    ublas::matrix_range<ublas::matrix<double> >
        Hhat_r(Hhat, ublas::range(0,n+1), ublas::range(0,n));
    Hhat_r = H + lambda*B;
    if (s < n) {
      Hhat_r = ublas::matrix_range<ublas::matrix<double> >
          (Hhat, ublas::range(n+1,2*n+1-s), ublas::range (0,n));
      Hhat_r = lambda*C;
    }
    ublas::vector<double> QR;
    factorQR(2*n-s+1, n, Hhat, QR);
    ublas::vector<double> rhs(2*n-s+1);
    applyQ(2*n-s+1, n, QR, g, rhs, true);
    solveR(2*n-s+1, n, QR, rhs, y);
    double norm_y = 0.0;
    for (int i = 0; i < n; i++)
      norm_y += y(i)*y(i);
    norm_y = sqrt(norm_y);    
    cout << norm_y << " ";
  }
  cout << "]" << endl;
  throw(-1);
#endif    
  
  for (int l = 0; l < maxNewt; l++) {
    // compute QR factorization of Hhat = [H + lambda*B; lambda*C]
    ublas::matrix_range<ublas::matrix<double> >
        Hhat_r(Hhat, ublas::range(0,n+1), ublas::range(0,n));
    Hhat_r = H + lambda*B;
    if (s < n) {
      Hhat_r = ublas::matrix_range<ublas::matrix<double> >
          (Hhat, ublas::range(n+1,2*n+1-s), ublas::range (0,n));
      Hhat_r = lambda*C;
    }
    ublas::vector<double> QR;
    factorQR(2*n-s+1, n, Hhat, QR);

    // TEMP: compute the SVD of Hhat, to see if the Sigma are close to zero
    computeSVD(2*n-s+1, n, Hhat, Sigma, P, UT);
    cout << "Hhat sigma = ";
    for (int i = 0; i < n; i++)
      cout << Sigma[i] << " ";
    cout << endl;
    
    // solve argmin || g - Hhat y ||
    ublas::vector<double> rhs(2*n-s+1);
    applyQ(2*n-s+1, n, QR, g, rhs, true);
    cout << "rhs = " << rhs(0) << ", " << rhs(1) << endl;
    solveR(2*n-s+1, n, QR, rhs, y);
    cout << "QR = " << QR(0) << ", " << QR(1) << endl;

    // check that y lies on the trust region; if so, exit
    double norm_y = 0.0;
    for (int i = 0; i < n; i++)
      norm_y += y(i)*y(i);
    norm_y = sqrt(norm_y);    
    cout << "\t\tsolveTrustReduced: Newton iter = " << l << ": res = "
         << fabs(norm_y - radius) << ": lambda = " << lambda << endl;
    if (fabs(norm_y -radius) < tol) {
      res = 0.0;
      for (int i = n; i < 2*n-s+1; i++)
        res += rhs(i)*rhs(i);
      res = sqrt(res);
      cout << "\t\tsolveTrustReduced: Newton converged with lambda = "
           << lambda << endl;
      return;
    }

    // update lambda

    // compute y*dy/dlambda in several steps...
    ublas::vector<double> w(n), z(n), v(n), r(2*n-s+1);
    // Step 1: solve R^T w = y
    solveR(2*n-s+1, n, QR, y, w, true);
    for (int i = 0; i < n; i++)
      rhs(i) = 0.0;
    // Step 2: solve R^T v = (dHhat/dlam)^T r (Note: (dHhat/dlam) = [B; C])
    applyQ(2*n-s+1, n, QR, rhs, r); // computes the residual
    if (s < n)
      Hhat_r = C;
    Hhat_r = ublas::matrix_range<ublas::matrix<double> >
        (Hhat, ublas::range(0,n+1), ublas::range(0,n));
    Hhat_r = B;
    solveR(2*n-s+1, n, QR, prod(trans(Hhat), r), v, true);
    // Step 3: compute z = Q^T (dHhat/dlam) y
    ublas::vector_range<ublas::vector<double> > y_r(y, ublas::range(0,n));
    applyQ(2*n-s+1, n, QR, prod(Hhat, y_r), r, true);
    // Step 4: compute z := v - z
    z = v - ublas::vector_range<ublas::vector<double> >(r, ublas::range(0,n));
    double dlam = inner_prod(w, z);

    // Step 2: compute the Newton update, and then safe-guard lambda
    dlam = (radius - norm_y)*norm_y*norm_y/(radius*dlam);
    //lambda += dlam;
    if (dlam < 0.0)
      lambda = std::max(lambda + dlam, std::max(-1.0000001*mineig, 0.0));
    else
      lambda = lambda + dlam; //std::min(lambda + dlam, 1e+3); // !!!! need better guard here, 10.0*lambda);
    //if (l == 2) break;
  }
  
  // if we get here, Newton's method failed to converge
  cerr << "krylov.cpp (solveTrustReduced): "
       << "Newton's method failed to converge to a valid lambda" << endl;
  throw(-1);
  
}
#endif
// ======================================================================

void solveTrustReduced(const int & n, const ublas::matrix<double> & H,
                       const double & radius, const ublas::vector<double> & g,
                       ublas::vector<double> & y, double & lambda,
                       double & pred) {
  if ( n < 0 ) {
    cerr << "krylov.cpp (solveTrustReduced): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if (radius < 0.0) {
    cerr << "krylov.cpp (solveTrustReduced): "
         << " trust-region radius must be nonnegative: radius = " << radius
         << endl;
    throw(-1);
  }
  if ( (H.size1() < n) || (H.size2() < n) ) {
    cerr << "krylov.cpp (solveTrustReduced): "
         << "H matrix dimension(s) smaller than necessary." << endl;
    throw(-1);
  }
  if ( (g.size() < n) || (y.size() < n) ) {
    cerr << "krylov.cpp (solveTrustReduced): "
         << "g and/or y vectors are smaller than necessary." << endl;
    throw(-1);
  }

  ublas::vector<double> eig(n, 0.0), y_tmp(n, 0.0);
  eigenvalues(n, H, eig);
  double eigmin = eig(0);  
#ifdef DEBUG
  cout << "\t\tsolveTrustReduced: eig = ";
  for (int j = 0; j < n; j++)
    cout << eig(j) << " ";
  cout << endl;
#endif
  
  //ublas::vector<double> UTU, work(n);
  //if (eigmin > 0.0) {
  lambda = 0.0;
  if (eigmin > 1e-12) { //sqrt(kEpsilon)) {
    // Hessian is semi-definite on span(Z), so solve for y and check if ||y|| is
    // in trust region radius
    double fnc, dfnc;
    boost::tie(y_tmp, fnc, dfnc) = trustFunction(n, H, g, lambda, radius);
    if (fnc < 0.0) { // i.e. norm_2(y) < raidus
      // compute predicted decrease in objective
      pred = 0.0;
      for (int i = 0; i < n; i++) y(i) = y_tmp(i);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
          pred -= 0.5*y(i)*H(i,j)*y(j);
        pred -= g(i)*y(i);
      }
      //pred = rhs(0)*y(0) - 0.5*pred;
      return;
    }
#ifdef DEBUG
    cout << "\t\tsolveTrustReduced: norm_2(y) = " << norm_2(y_tmp) << " > "
         << radius << endl;
#endif
  }
  // if we get here, either the Hessian is semi-definite or ||y|| > radius

  // bracket the Lagrange multiplier lambda
  int max_brk = 20;
  double dlam = 0.1*std::max(-eigmin, kEpsilon);
  double lambda_h = std::max(-eigmin, 0.0) + dlam;
  double fnc_h, dfnc;
  boost::tie(y_tmp, fnc_h, dfnc) = trustFunction(n, H, g, lambda_h, radius);
  for (int k = 0; k < max_brk; k++) {
#ifdef VERBOSE_DEBUG
    cout << "\t\tsolveTrustReduced: (lambda_h, fnc_h) = ("
         << lambda_h << ", " << fnc_h << ")" << endl;
#endif
    if (fnc_h > 0.0) break;
    dlam *= 0.1;
    lambda_h = std::max(-eigmin, 0.0) + dlam;
    boost::tie(y_tmp, fnc_h, dfnc) = trustFunction(n, H, g, lambda_h, radius);
  }
  dlam = sqrt(kEpsilon);
  double lambda_l = std::max(-eigmin, 0.0) + dlam;
  double fnc_l;
  boost::tie(y_tmp, fnc_l, dfnc) = trustFunction(n, H, g, lambda_l, radius);
  for (int k = 0; k < max_brk; k++) {
#ifdef VERBOSE_DEBUG
    cout << "\t\tsolveTrustReduced: (lambda_l, fnc_l) = ("
         << lambda_l << ", " << fnc_l << ")" << endl;
#endif
    if (fnc_l < 0.0) break;
    dlam *= 100.0;
    lambda_l = std::max(-eigmin, 0.0) + dlam;
    boost::tie(y_tmp, fnc_l, dfnc) = trustFunction(n, H, g, lambda_l, radius);
  }  
  lambda = 0.5*(lambda_l + lambda_h);
#ifdef VERBOSE_DEBUG
  cout << "\t\tsolveTrustReduced: initial lambda = " << lambda << endl;
#endif
  
  // Apply (safe-guarded) Newton's method to find lambda
  double dlam_old = fabs(lambda_h - lambda_l);
  dlam = dlam_old;

  int maxNewt = 50;
  double tol = sqrt(kEpsilon);
  double lam_tol = sqrt(kEpsilon)*dlam;
  double fnc;
  boost::tie(y_tmp, fnc, dfnc) = trustFunction(n, H, g, lambda, radius);
  double res0 = fabs(fnc);
  int l;
  for (l = 0; l < maxNewt; l++) {

    // check if y lies on the trust region; if so, exit
#ifdef VERBOSE_DEBUG
    cout << "\t\tsolveTrustReduced: Newton iter = " << l << ": res = "
         << fabs(fnc) << ": lambda = " << lambda << endl;
#endif
    if ( (fabs(fnc) < tol*res0) || (fabs(dlam) < lam_tol) ) {
#ifdef DEBUG
      cout << "\t\tsolveTrustReduced: Newton converged with lambda = "
           << lambda << endl;      
#endif
      break;
    }

    // choose safe-guarded step
    if ( ( ((lambda - lambda_h)*dfnc - fnc)*
           ((lambda - lambda_l)*dfnc - fnc) > 0.0) ||
         (fabs(2.0*fnc) > fabs(dlam_old*dfnc) ) ) {
      // use bisection if Newton-step is out of range or not decreasing fast
      dlam_old = dlam;
      dlam = 0.5*(lambda_h - lambda_l);
      lambda = lambda_l + dlam;
      if (lambda_l == lambda) break;
    } else {
      // Newton-step is acceptable
      dlam_old = dlam;
      dlam = fnc/dfnc;
      double temp = lambda;
      lambda -= dlam;
      if (temp == lambda) break;
    }

    // evaluate function at new lambda
    boost::tie(y_tmp, fnc, dfnc) = trustFunction(n, H, g, lambda, radius);
    if (fnc < 0.0)
      lambda_l = lambda;
    else
      lambda_h = lambda;
  }
  if (l == maxNewt) {
    // Newton's method failed to converge
    cerr << "krylov.cpp (solveTrustReduced): "
         << "Newton's method failed to converge to a valid lambda" << endl;
    throw(-1);
  }       
  // compute predicted decrease in objective
  pred = 0.0;
  for (int i = 0; i < n; i++) y(i) = y_tmp(i);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      pred -= 0.5*y(i)*H(i,j)*y(j);
    pred -= g(i)*y(i);
  }
}
// ======================================================================
boost::tuple<ublas::vector<double>, double, double> 
trustFunction(const int& n, const ublas::matrix<double> & H,
              const ublas::vector<double> & g, const double& lambda,
              const double& radius) {
  // First, factorize the matrix [H + lambda*I], where H is the reduced Hessain
  double diag = std::max(1.0, lambda)*0.01*kEpsilon;
  bool semidefinite = true;
  int regiter = 0;
  ublas::matrix<double> Hhat(n,n);
  ublas::vector<double> UTU, work(n);
  while (semidefinite) {
    ++regiter;
    try {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
          Hhat(i,j) = H(i,j);
        Hhat(i,i) += lambda + diag;
      }
      factorCholesky(n, Hhat, UTU);
      semidefinite = false;
    } catch (bool factor_failed) {
      diag *= 100.0;
#ifdef VERBOSE_DEBUG
      cout << "\t\ttrustFunction: factorCholesky() failed,"
           << " adding " << diag << " to diagonal..." << endl;
#endif
    }
    if (regiter > 20) {
      cerr << "krylov.cpp (trustFunction): "
           << "regularization of Cholesky factorization failed." << endl;
      throw(-1);
    }
  }

  // Next, solve for the step; the step's length is used to define the objective
  ublas::vector<double> y(n);
  solveU(n, UTU, g, work, true);
  solveU(n, UTU, work, y, false);
  y *= -1.0; // to "move" g to rhs

  // compute the function
  double norm_y = norm_2(y);  
  double fnc = 1.0/radius - 1.0/norm_y;
  
  // find derivative of the function
  solveU(n, UTU, y, work, true);
  double norm_work = norm_2(work);
  double dfnc = norm_work/norm_y;
  dfnc = -(dfnc*dfnc)/norm_y;

  // return step, function, and derivative
  return boost::make_tuple(y, fnc, dfnc);
}

// ======================================================================

void solveUnderdeterminedMinNorm(const int & nrow, const int & ncol,
                                 const ublas::matrix<double> & A,
                                 const ublas::vector<double> & b,
                                 ublas::vector<double> & x) {
  if ( ( nrow < 0 ) || (ncol < 0) ) {
    cerr << "krylov.cpp (solveUnderdeterminedMinNorm): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if (nrow > ncol) {
    cerr << "krylov.cpp (solveUnderdeterminedMinNorm): "
	 << "expecting rectangular matrix with nrow <= ncol." << endl;
    throw(-1);
  }
  if ( (A.size1() < nrow) || (A.size2() < ncol) ) {
    cerr << "krylov.cpp (solveUnderdeterminedMinNorm): "
	 << "A matrix sizes inconsistent with nrow and ncol." << endl;
    throw(-1);
  }
  if ( (b.size() < nrow) || (x.size() < ncol) ) {
    cerr << "krylov.cpp (solveUnderdeterminedMinNorm): "
         << "b and/or x vectors are smaller than necessary." << endl;
    throw(-1);
  }

  // peform SVD of A^{T}
  ublas::vector<double> Sigma, P, QT;
  computeSVD(ncol, nrow, ublas::trans(A), Sigma, P, QT, true);
  ublas::matrix<double> PTmat(ncol, ncol, 0.0), Qmat(nrow, nrow, 0.0);
  for (int k = 0; k < ncol; k++)
    for (int j = 0; j < ncol; j++)
      PTmat(k,j) = P(k*ncol+j);
  for (int k = 0; k < nrow; k++)
    for (int j = 0; j < nrow; j++)
      Qmat(k,j) = QT(k*nrow+j);
  solveUnderdeterminedMinNorm(nrow, ncol, Sigma, Qmat, PTmat, b, x);
}

// ======================================================================

void solveUnderdeterminedMinNorm(const int & nrow, const int & ncol,
                                 const ublas::vector<double> & Sigma,
                                 const ublas::matrix<double> & U,
                                 const ublas::matrix<double> & VT,
                                 const ublas::vector<double> & b,
                                 ublas::vector<double> & x) {
  if ( ( nrow < 0 ) || (ncol < 0) ) {
    cerr << "krylov.cpp (solveUnderdeterminedMinNorm): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if (nrow > ncol) {
    cerr << "krylov.cpp (solveUnderdeterminedMinNorm): "
	 << "expecting rectangular matrix with nrow <= ncol." << endl;
    throw(-1);
  }
  if ( (U.size1() < nrow) || (U.size2() < nrow) ||
       (VT.size1() < nrow) || (VT.size2() < ncol) ||
       (Sigma.size() < nrow) ) {
    cerr << "krylov.cpp (solveUnderdeterminedMinNorm): "
         << "matrices of left or right singular vectors, or Sigma matrix, "
         << "are smaller than necessary." << endl;
    throw(-1);
  }
  if ( (b.size() < nrow) || (x.size() < ncol) ) {
    cerr << "krylov.cpp (solveUnderdeterminedMinNorm): "
         << "b and/or x vectors are smaller than necessary." << endl;
    throw(-1);
  }

  // compute rank of A = U*Sigma*VT
  int rank = 0;
  for (int i = 0; i < nrow; i++)
    if (Sigma(i) > 0.01*Sigma(0)) rank++; //kEpsilon*Sigma(0)) rank++;

  if ( (rank == 0) || (Sigma(0) < kEpsilon) ) {
    cerr << "krylov.cpp (solveUnderdeterminedMinNorm): "
         << "singular values are all zero or negative." << endl;
    throw(-1);
  }
  ublas::vector<double> y(rank, 0.0);
  for (int i = 0; i < rank; i++) {
    y(i) = 0.0;
    for (int j = 0; j < nrow; j++)
      y(i) += U(j,i)*b(j);
    y(i) /= Sigma(i);
  }
  for (int j = 0; j < ncol; j++) {
    x(j) = 0.0;
    for (int i = 0; i < rank; i++)
      x(j) += y(i)*VT(i,j);
  }
}

// ======================================================================

void solveLeastSquares(const int & nrow, const int & ncol,
                       const ublas::matrix<double>& A,
                       const ublas::vector<double>& b,
                       ublas::vector<double>& x) {
  if ( ( nrow < 0 ) || (ncol < 0) ) {
    cerr << "krylov.cpp (solveLeastSquares): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if (nrow < ncol) {
    cerr << "krylov.cpp (solveLeastSquares): "
	 << "expecting rectangular matrix with nrow >= ncol." << endl;
    throw(-1);
  }
  // copy A into Awrk in column major ordering
  ublas::vector<double> Awrk(nrow*ncol, 0.0);
  for (int j = 0; j < ncol; j++)
    for (int i = 0; i < nrow; i++)
      Awrk(j*nrow + i) = A(i,j);
  char trans = 'N';
  int m = nrow;
  int n = ncol;
  int nrhs = 1;
  ublas::vector<double> rhs(b);
  int lwork = ncol + nrow;
  ublas::vector<double> work(lwork, 0.0);
  int info;
  dgels_(&trans, &m, &n, &nrhs, &*Awrk.begin(), &m, &*rhs.begin(),
         &m, &*work.begin(), &lwork, &info);
  if (info != 0) {
    cerr << "krylov.cpp (solveLeastSquares): "
	 << "LAPACK routine dgels failed with info = " << info << endl;
    throw(-1);
  }
  for (int i = 0; i < ncol; i++) x(i) = rhs(i);
}

// ======================================================================

double solveLeastSquaresOverSphere(const int & nrow, const int & ncol,
                                   const double & radius,
                                   const ublas::vector<double> & Sigma,
                                   const ublas::matrix<double> & U, 
                                   const ublas::matrix<double> & VT,
                                   const ublas::vector<double> & b,
                                   ublas::vector<double> & x) {
  if ( ( nrow < 0 ) || (ncol < 0) ) {
    cerr << "krylov.cpp (solveLeastSquaresOverSphere): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if (nrow < ncol) {
    cerr << "krylov.cpp (solveLeastSquaresOverSphere): "
	 << "expecting rectangular matrix with nrow >= ncol." << endl;
    throw(-1);
  }
  if (radius < 0.0) {
    cerr << "krylov.cpp (solveLeastSquaresOverSphere): "
         << " sphere radius must be nonnegative: radius = " << radius
         << endl;
    throw(-1);
  }
  if ( (U.size1() < nrow) || (U.size2() < ncol) ||
       (VT.size1() < ncol) || (VT.size2() < ncol) ||
       (Sigma.size() < ncol) ) {
    cerr << "krylov.cpp (solveLeastSquaresOverSphere): "
         << "matrices of left or right singular vectors, or Sigma matrix, "
         << "are smaller than necessary." << endl;
    throw(-1);
  }
  if ( (b.size() < nrow) || (x.size() < ncol) ) {
    cerr << "krylov.cpp (solveLeastSquaresOverSphere): "
         << "b and/or x vectors are smaller than necessary." << endl;
    throw(-1);
  }

  // compute rank of A = U*Sigma*VT
  int rank = 0;
  for (int i = 0; i < ncol; i++)
    if (Sigma(i) > kEpsilon*Sigma(0)) rank++;
  if ( (rank == 0) || (Sigma(0) < kEpsilon) ) {
    cerr << "krylov.cpp (solveLeastSquaresOverSphere): "
         << "singular values are all zero or negative." << endl;
    throw(-1);
  }

  // compute the tentative reduced-space solution, y = Sigma^{-1} U^T b
  ublas::vector<double> g(rank, 0.0), y(rank, 0.0);
  double sol_norm = 0.0;
  for (int i = 0; i < rank; i++) {
    for (int j = 0; j < nrow; j++)
      g(i) += U(j,i)*b(j);
    y(i) = g(i)/Sigma(i);
    sol_norm += y(i)*y(i);
  }

  double rad2 = radius*radius;
  double lambda = 0.0;
  if (sol_norm > rad2) {
    // tentative solution is outside sphere, so solve secular equation
    double phi = sol_norm - rad2;
    double phi0 = phi;
    const int kMaxIter = 40;
    const double kTol = 100.0*kEpsilon;
    int k;
    for (k = 0; k < kMaxIter; k++) {
      cout << "iter " << k << ": res = " << fabs(phi) << endl;
      if (fabs(phi) < kTol*phi0) break;
      double dphi = 0.0;
      for (int i = 0; i < rank; i++)
        dphi -= 2.0*pow(Sigma(i)*g(i), 2.0)/pow(Sigma(i)*Sigma(i) + lambda, 3.0);
      lambda -= phi/dphi;
      sol_norm = 0.0;
      for (int i = 0; i < rank; i++) {
        y(i) = Sigma(i)*g(i)/(Sigma(i)*Sigma(i) + lambda);
        sol_norm += y(i)*y(i);
      }
      phi = sol_norm - rad2;
    }
    if (k == kMaxIter) {
      cerr << "krylov.cpp (solveLeastSquaresOverSphere): "
           << "maximum number of Newton iterations exceeded." << endl;
      cout << "fabs(phi) = " << fabs(phi) << endl;
      cout << "kTol*phi0 = " << kTol*phi0 << endl;
      throw(-1);
    }
  }
  for (int j = 0; j < ncol; j++) {
    x(j) = 0.0;
    for (int i = 0; i < rank; i++)
      x(j) += y(i)*VT(i,j);
  }
  return lambda;
}

// ======================================================================

double trustResidual(const int & n, const ublas::matrix<double> & H,
                     const ublas::matrix<double> & B,
                     const ublas::vector<double> & g,
                     const ublas::vector<double> & y,
                     const double & lambda) {
  if ( n < 0 ) {
    cerr << "krylov.cpp (trustResidual): "
	 << "matrix dimensions must be greater than 0." << endl;
    throw(-1);
  }
  if ( (H.size1() < n+1) || (H.size2() < n) ||
       (B.size1() < n+1) || (B.size2() < n) ) {
    cerr << "krylov.cpp (trustResidual): "
         << "H and/or B matrix are smaller than necessary." << endl;
    throw(-1);
  }
  if ( (g.size() < 2*n+1) || (y.size() < n) ) {
    cerr << "krylov.cpp (trustResidual): "
         << "g and/or y vectors are smaller than necessary." << endl;
    throw(-1);
  }
  
  // find SVD of B, and find index s such that
  //     1 = Sigma(0) = --- = Sigma(s-1) > Sigma(s)  
  // Note: for the given B, the Sigma should lie between 0 and 1 (see pg 603,
  // Golub and van Loan); check this here too
  ublas::vector<double> Sigma, P, UT;
  computeSVD(n+1, n, B, Sigma, P, UT);
  for (int s = 0; s < n; s++) {
    if ( (Sigma(s) < - 1E+4*kEpsilon) || (Sigma(s) > 1.0+1E+4*kEpsilon) ) {
      cerr << "krylov.cpp (trustResidual): "
           << "Singular values of B are not in [0,1]. "
           << "check that Z and V have orthogonal columns" << endl;
      cout << "B sigma = ";
      for (int i = 0; i < n; i++)
        cout << Sigma[i] << " ";
      cout << endl;
      throw(-1);
    }
  }

  int s;
  for (s = 0; s < n; s++)
    if (fabs(Sigma(s) - 1.0) > sqrt(kEpsilon)) break; //1E+4*kEpsilon) break;
  
#ifdef DEBUG
  cout << "\t\ttrustResidual: singular values = ";
  for (int i = 0; i < n; i++)
    cout << std::setw(16) << std::setprecision(12) << Sigma(i) << " ";    
  cout << endl;
  cout << "\t\ttrustResidual: n, s, n-s = " << n << ", " << s << ", "
       << n-s << endl;
#endif
  
  // compute the matrix C = W^T * Z = N*UT(s:n-1)*(I - B^T*B)
  // where N^{-1} = diag(sqrt(1 - Sigma(s:n-1)))
  ublas::matrix<double> C;
  if (s < n) {
    C.resize(n-s,n);
    // Step 1: compute P := I - B^T*B (stored column major)
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        P(i+j*n) = 0.0;
        for (int k = 0; k < n+1; k++)
          P(i+j*n) -= B(k,i)*B(k,j);
      }
      P(j+j*n) += 1.0;
    }
    // Step 2: compute C := QT(s:n-1)*P
    for (int i = 0; i < n-s; i++) {
      for (int j = 0; j < n; j++) {
        C(i,j) = 0.0;
        for (int k = 0; k < n; k++)
          C(i,j) += UT(s+i+k*n)*P(k+j*n);
      }
    }
    // Step 3: compute C := N*C
    for (int i = 0; i < n-s; i++) {
      double fac = 1.0/sqrt(1.0 - Sigma(s+i)*Sigma(s+i));
      for (int j = 0; j < n; j++)
        C(i,j) *= fac;
    }
  }

  //cout << "trustResidual: after computing C..." << endl;

  // compute reduced residual
  ublas::vector<double> r(2*n+1-s);

  double res_norm = 0.0;
  for (int i = 0; i < n+1; i++) {
    double tmp = g(i);
    for (int j = 0; j < n; j++)
      tmp -= (H(i,j) + lambda*B(i,j))*y(j);
    res_norm += tmp*tmp;
  }
  for (int i = 0; i < n-s; i++) {
    double tmp = g(n+1+i);
    for (int j = 0; j < n; j++)
      tmp -= lambda*C(i,j)*y(j);
    res_norm += tmp*tmp;
  }
  return sqrt(res_norm);
}

// ======================================================================

bool triMatixInvertible(const int & n, const ublas::matrix<double> & T) {
  for (int i = 0; i < n; i++)
    if (fabs(T(i,i)) < kEpsilon) return false;
  return true;
}

// ======================================================================

void writeKrylovHeader(ostream & os, const std::string & solver,
		       const double & restol, const double & resinit) {
#if 0
  if (!(os.good())) {
    cerr << "krylov.h (writeKrylovHeader): "
	 << "ostream is not good for i/o operations." << endl;
    throw(-1);
  }
#endif
  std::ios_base::fmtflags old = os.setf(ios::fixed, ios::floatfield);
  os << "# " << solver << " residual history" << endl;
  os << "# " << "residual tolerance target = "
     << std::setw(10) << std::setprecision(6) << restol << endl;
  os << "# " << "initial residual norm     = "
     << std::setw(10) << std::setprecision(6) << resinit << endl;
  os << std::setw(6) << "# iter" << std::setw(12) << "rel. res." << endl;
  os.setf(old, ios::floatfield);
}

// ======================================================================

void writeKrylovHeader(ostream & os, const std::string & solver,
		       const double & restol, const double & resinit,
                       const boost::format & col_header) {
#if 0
  if (!(os.good())) {
    cerr << "krylov.cpp (writeKrylovHeader): "
	 << "ostream is not good for i/o operations." << endl;
    throw(-1);
  }
#endif
  os << boost::format(
      "# solver %|s| convergence history\n") % solver;
  os << boost::format(
      "# residual tolerance target %|30t| = %|-10.6|\n") % restol;
  os << boost::format(
      "# initial residual norm %|30t| = %|-10.6|\n") % resinit;
  os << col_header;
}

// ======================================================================

void writeKrylovHistory(ostream & os, const int & iter,
			const double & res, const double & resinit) {
#if 0
  if (!(os.good())) {
    cerr << "krylov.h (writeKrylovHistory): "
	 << "ostream is not good for i/o operations." << endl;
    throw(-1);
  }
#endif
  std::ios_base::fmtflags old = os.setf(ios::scientific, ios::floatfield);
  os << std::setw(5) << iter
     << std::setw(15) << std::setprecision(6) << res/resinit
     << endl;
  os.setf(old, ios::floatfield);
}

} // namespace kona
