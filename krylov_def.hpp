/**
 * \file krylov_def.hpp
 * \brief implementations of templated Krylov-subspace methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 * \version 1.0
 */

#pragma once

//#include <boost/numeric/ublas/zero_vector.hpp>
//#include <boost/numeric/ublas/zero_matrix.hpp>
#include <boost/lexical_cast.hpp>

namespace kona {
// ==============================================================================
template <class Vec>
FGMRESSolver<Vec>::FGMRESSolver() {
  maxiter_ = -1;
}
// ==============================================================================
template <class Vec>
void FGMRESSolver<Vec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "FMGRESSolver::SubspaceSize(): "
         << "illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  maxiter_ = m;

  // Note: STL vectors are used for W_ and Z_ while boost::numeric::ublas vectors
  // are used for double based vectors and matrices; this choice was made
  // mostly to help emphasize that the objects in W_ and Z_ are fundamentally
  // different (they are some sort of vectors themselves).
  W_.reserve(maxiter_+1);
  Z_.reserve(maxiter_);
  g_.resize(maxiter_+1, 0.0);
  sn_.resize(maxiter_+1, 0.0);
  cs_.resize(maxiter_+1, 0.0);
  y_.resize(maxiter_, 0.0);
  H_.resize(maxiter_+1, maxiter_, 0.0);
}
// ==============================================================================
template <class Vec>
void FGMRESSolver<Vec>::MemoryRequired(ptree& num_required) const {
  if (maxiter_ < 1) {
    cerr << "FMGRESSolver::MemoryRequired(): "
         << "SubspaceSize must be called first (maxiter_ is undefined)" << endl;
    throw(-1);
  }
  num_required.put("num_vec", 2*maxiter_ + 1);
}
// ==============================================================================
template <class Vec>
void FGMRESSolver<Vec>::Solve(const ptree & ptin, const Vec & b, Vec & x,
                              MatrixVectorProduct<Vec,Vec> & mat_vec,
                              Preconditioner<Vec,Vec> & precond,
                              ptree & ptout, ostream & fout) {
  W_.clear();
  Z_.clear();
  g_ = ublas::zero_vector<double>(g_.size());
  sn_ = ublas::zero_vector<double>(sn_.size());
  cs_ = ublas::zero_vector<double>(cs_.size());
  y_ = ublas::zero_vector<double>(y_.size());
  H_ = ublas::zero_matrix<double>(H_.size1(),H_.size2());
  int iters = 0;
  
  // calculate the norm of the rhs vector
  double norm0 = b.Norm2();

  // calculate the initial residual (actually the negative residual)
  // and compute its norm
  if (ptin.get<bool>("dynamic", false))
    mat_vec.set_product_tol(ptin.get<double>("tol")/
                            static_cast<double>(maxiter_));
  W_.push_back(b); // needed to initialize W_.[0], unfortunately
  mat_vec(x, W_[0]);
  W_[0] -= b;

  double beta = W_[0].Norm2();
  if ( (beta < ptin.get<double>("tol")*norm0) || (beta < kEpsilon) ) {
    // system is already solved
    fout << "FGMRES system solved by initial guess." << endl;
    ptout.put<int>("iters", iters);
    ptout.put<double>("res", beta);
    return;
  }

  // normalize residual to get w_{0} (the negative sign is because W_[0]
  // holds the negative residual, as mentioned above)
  W_[0] /= -beta;

  // initialize the RHS of the reduced system
  g_(0) = beta;

  // output header information including initial residual
  int i = 0;
  writeKrylovHeader(fout, "FGMRES", ptin.get<double>("tol"), beta);
  writeKrylovHistory(fout, i, beta, norm0);

  // loop over all search directions
  bool lin_depend = false;
  for (i = 0; i < maxiter_; i++) {
    // check if solution has converged; also, if we have a w vector
    // that is linearly dependent, check that we have converged
    if ( (lin_depend) && (beta > ptin.get<double>("tol")*norm0) ) {
      cerr << "FGMRES: Arnoldi process breakdown: "
           << "H(" << i+1 << "," << i << ") = " << H_(i+1,i)
           << ", however ||res|| = " << beta << endl;
      throw(-1);
    } else if (beta < ptin.get<double>("tol")*norm0) break;
    iters++;

    // precondition the Vec W_[i] and store result in Z_[i]
    Z_.push_back(b);
    precond(W_[i], Z_[i]);

    // add to Krylov subspace
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
                              (beta*static_cast<double>(maxiter_)));
    W_.push_back(Z_[i]);
    mat_vec(Z_[i], W_[i+1]);

    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H_, W_);
    } catch (string err_msg) {
      cerr << "FGMRES: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

    // apply old Givens rotations to new column of the Hessenberg matrix
    // then generate the new Givens rotation matrix and apply it to
    // the last two elements of H(i,:) and g
    for (int k = 0; k < i; k++)
      applyGivens(sn_(k), cs_(k), H_(k,i), H_(k+1,i));
    generateGivens(H_(i,i), H_(i+1,i), sn_(i), cs_(i));
    applyGivens(sn_(i), cs_(i), g_(i), g_(i+1));

    // set L2 norm of residual and output the relative residual if necessary
    beta = fabs(g_(i+1));
    writeKrylovHistory(fout, i+1, beta, norm0);
  }

  // solve the least-squares system and update solution
  solveReducedHessenberg(i, H_, g_, y_);
  for (int k = 0; k < i; k++) {
    // x = x + y[k]*Z_[k]
    x.EqualsAXPlusBY(1.0, x, y_[k], Z_[k]);
  }

  ptout.put<int>("iters", iters);
  ptout.put<double>("res", beta);
  if (ptin.get<bool>("return sig",false)) {
    // compute smallest singlular value of H
    ublas::vector<double> Sigma, P, UT;
    computeSVD(i+1, i, H_, Sigma, P, UT);
    cout << "[ j singular value of H = " << Sigma(i-1) << endl;
  }
  
  if (ptin.get<bool>("check", false)) {
    // recalculate explicilty and check final residual
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(0.1*ptin.get<double>("tol")/
                              static_cast<double>(maxiter_));
    mat_vec(x, W_[0]);
    // W_[0] = b - W_[0] = b - A*x
    W_[0].EqualsAXPlusBY(1.0, b, -1.0, W_[0]);
    double true_res = W_[0].Norm2();
    fout << "# FGMRES final (true) residual : |res|/|res0| = "
         << true_res/norm0 << endl;
    ptout.put<double>("res", true_res);
    if (fabs(true_res - beta) > 0.01*ptin.get<double>("tol")*norm0) {
      fout << "# WARNING in FGMRES: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = " << (true_res - beta)/norm0 << endl;
    }
  }
}
// ==============================================================================
template <class Vec>
FFOMSolver<Vec>::FFOMSolver() {
  maxiter_ = -1;
}
// ==============================================================================
template <class Vec>
void FFOMSolver<Vec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "FFOMSolver::SubspaceSize(): "
         << "illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  maxiter_ = m;

  // Note: STL vectors are used for W_ and Z_ while boost::numeric::ublas vectors
  // are used for double based vectors and matrices; this choice was made
  // mostly to help emphasize that the objects in W_ and Z_ are fundamentally
  // different (they are some sort of vectors themselves).
  W_.reserve(maxiter_+1);
  Z_.reserve(maxiter_);
  g_.resize(maxiter_+1, 0.0);
  sn_.resize(maxiter_+1, 0.0);
  cs_.resize(maxiter_+1, 0.0);
  y_.resize(maxiter_, 0.0);
  y_old_.resize(maxiter_, 0.0);
  H_.resize(maxiter_+1, maxiter_, 0.0);
}
// ==============================================================================
template <class Vec>
void FFOMSolver<Vec>::MemoryRequired(ptree& num_required) const {
  if (maxiter_ < 1) {
    cerr << "FFOMSolver::MemoryRequired(): "
         << "SubspaceSize must be called first (maxiter_ is undefined)" << endl;
    throw(-1);
  }
  num_required.put("num_vec", 2*maxiter_ + 1);
}
// ==============================================================================
template <class Vec>
void FFOMSolver<Vec>::Solve(const ptree & ptin, const Vec & b, Vec & x,
                            MatrixVectorProduct<Vec,Vec> & mat_vec,
                            Preconditioner<Vec,Vec> & precond,
                            ptree & ptout, ostream & fout) {
  W_.clear();
  Z_.clear();  
  g_ = ublas::zero_vector<double>(g_.size());
  sn_ = ublas::zero_vector<double>(sn_.size());
  cs_ = ublas::zero_vector<double>(cs_.size());
  y_ = ublas::zero_vector<double>(y_.size());
  y_old_ = ublas::zero_vector<double>(y_old_.size());
  H_ = ublas::zero_matrix<double>(H_.size1(),H_.size2());
  int iters = 0;
  
  // calculate the norm of the rhs vector
  double norm0 = b.Norm2();

  // calculate the initial residual (actually the negative residual)
  // and compute its norm
  if (ptin.get<bool>("dynamic", false))
    mat_vec.set_product_tol(ptin.get<double>("tol")/
                            static_cast<double>(maxiter_));
  W_.push_back(b); // needed to initialize W_.[0], unfortunately
  mat_vec(x, W_[0]);
  W_[0] -= b;

  double beta = W_[0].Norm2();
  if ( (beta < ptin.get<double>("tol")*norm0) || (beta < kEpsilon) ) {
    // system is already solved
    fout << "FFOM system solved by initial guess." << endl;
    ptout.put<int>("iters", iters);
    ptout.put<double>("res", beta);
    return;
  }

  // normalize residual to get w_{0} (the negative sign is because W_[0]
  // holds the negative residual, as mentioned above)
  W_[0] /= -beta;

  // initialize the RHS of the reduced system
  g_(0) = beta;

  // output header information including initial residual
  int i = 0;
  writeKrylovHeader(fout, "FFOM", ptin.get<double>("tol"), beta);
  writeKrylovHistory(fout, i, beta, norm0);

  // loop over all search directions
  bool lin_depend = false;
  double beta_old = norm0;
  for (i = 0; i < maxiter_; i++) {
    // check if solution has converged
    if (beta < ptin.get<double>("tol")*norm0) break;
    iters++;

    // precondition the Vec W_[i] and store result in Z_[i]
    Z_.push_back(b);
    precond(W_[i], Z_[i]);

    // add to Krylov subspace
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
                              (beta*static_cast<double>(maxiter_)));
    W_.push_back(Z_[i]);
    mat_vec(Z_[i], W_[i+1]);

    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H_, W_);
    } catch (string err_msg) {
      cerr << "FFOM: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

    // apply old Givens rotations to new column of the Hessenberg matrix
    for (int k = 0; k < i; k++)
      applyGivens(sn_(k), cs_(k), H_(k,i), H_(k+1,i));

    // Check if the reduced system is singular; if not, solve the reduced
    // square system and compute the new residual norm
    if (triMatixInvertible(i+1, H_)) {
      solveReducedHessenberg(i+1, H_, g_, y_);
      beta_old = beta;
      beta = fabs(y_(i))*H_(i+1,i);
    }
    writeKrylovHistory(fout, i+1, beta, norm0);

    // if we have a W_ vector that is linearly dependent, check that we have
    // converged
    if ( (lin_depend) && (beta > ptin.get<double>("tol")*norm0) ) {
      cerr << "FFOM: Arnoldi process breakdown: "
           << "H(" << i+1 << "," << i << ") = " << H_(i+1,i)
           << ", however ||res|| = " << beta << endl;
      throw(-1);
    } 
    
    // Generate the new Givens rotation matrix and apply it to
    // the last two elements of H(i,:) and g
    generateGivens(H_(i,i), H_(i+1,i), sn_(i), cs_(i));
    applyGivens(sn_(i), cs_(i), g_(i), g_(i+1));
  }

  // compute solution
  for (int k = 0; k < i; k++) {
    // x = x + y[k]*z[k]
    x.EqualsAXPlusBY(1.0, x, y_[k], Z_[k]);
  }

  ptout.put<int>("iters", iters);
  ptout.put<double>("res", beta);
  
  if (ptin.get<bool>("check", false)) {
    // recalculate explicilty and check final residual
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(0.1*ptin.get<double>("tol")/
                              static_cast<double>(maxiter_));
    mat_vec(x, W_[0]);
    // W_[0] = b - W_[0] = b - A*x
    W_[0].EqualsAXPlusBY(1.0, b, -1.0, W_[0]);
    double true_res = W_[0].Norm2();
    fout << "# FFOM final (true) residual : |res|/|res0| = "
         << true_res/norm0 << endl;
    ptout.put<double>("res", true_res);
    if (fabs(true_res - beta) > 0.01*ptin.get<double>("tol")*norm0) {
      fout << "# WARNING in FFOM: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = " << (true_res - beta)/norm0 << endl;
    }
  }
}
// ==============================================================================
template <class Vec>
MINRESSolver<Vec>::MINRESSolver() {
  maxiter_ = -1;
}
// ==============================================================================
template <class Vec>
void MINRESSolver<Vec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "MINRESSolver::SubspaceSize(): "
         << "illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  maxiter_ = m;
  work_.reserve(7);
}
// ==============================================================================
template <class Vec>
void MINRESSolver<Vec>::MemoryRequired(ptree& num_required) const {
  num_required.put("num_vec", 7);
}
// ==============================================================================
template <class Vec>
void MINRESSolver<Vec>::Solve(const ptree & ptin, const Vec & b, Vec & x,
                              MatrixVectorProduct<Vec,Vec> & mat_vec,
                              Preconditioner<Vec,Vec> & precond,
                              ptree & ptout, ostream & fout) {
  // allocate memory, and set some references for readability
  work_.resize(7, x);
  Vec& y = work_[0];
  Vec& w = work_[1];
  Vec& r1 = work_[2];
  Vec& r2 = work_[3];
  Vec& w1 = work_[4];
  Vec& w2 = work_[5];
  Vec& v = work_[6];

  double Anorm = 0.0;
  double Acond = 0.0;
  double rnorm = 0.0;
  double ynorm = 0.0;
  x = 0.0;

  // Set up y and v for the first Lanczos vector v1.
  // y = beta0 P' v1, where P = C**(-1).
  // v is really P' v1.
  w = b;
  precond(w, y);
  r1 = b;
  double beta0 = InnerProd(b, y);
  int iters = 0;

  if (beta0 < 0.0) {
    // preconditioner must be indefinite
    cerr << "MINRESSolver::Solve(): "
         << "preconditioner provided is indefinite." << endl;
    throw(-1);
  }
  if (beta0 == 0.0) {
    // rhs must be zero, so stop with x = 0
    fout << "MINRESSolver::Solve(): "
         << "rhs vector b = 0, so returning x = 0." << endl;
    ptout.put<int>("iters", iters);
    ptout.put<double>("res", 0.0);
    return;
  }  
  beta0 = sqrt(beta0);
  rnorm = beta0;
  
  // should provide checks here to see if preconditioner and matrix are
  // symmetric
  if (ptin.get<bool>("dynamic", false))
    mat_vec.set_product_tol(ptin.get<double>("tol")/
                            static_cast<double>(maxiter_));
  mat_vec(y, w);
  double Arnorm1 = w.Norm2();

  // initialized remaining quantities in preparation for iteration
  double oldb = 0.0;
  double beta = beta0;
  double dbar = 0.0;
  double oldeps = 0.0;
  double epsln = 0.0;
  double qrnorm = beta0;
  double phibar = beta0;
  double rhs1 = beta0;
  double rhs2 = 0.0;
  double tnorm2 = 0.0;
  double ynorm2 = 0.0;
  double cs = -1.0;
  double sn = 0.0;
  double gmax = 0.0;
  double gmin = 0.0;
  w2 = x; // recall x = 0
  w1 = x; // recall x = 0
  r2 = r1;

  // output header information including initial residual
  int i = 0;
  writeKrylovHeader(fout, "MINRES", ptin.get<double>("tol"), beta0);
  writeKrylovHistory(fout, i, beta0, beta0);

  // loop over all serach directions
  for (i = 0; i < maxiter_; i++) {

    // Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
    // The general iteration is similar to the case k = 1 with v0 = 0:
    //
    //   p1      = Operator * v1  -  beta0 * v0,
    //   alpha1  = v1'p1,
    //   q2      = p2  -  alpha1 * v1,
    //   beta2^2 = q2'q2,
    //   v2      = (1/beta2) q2.
    //
    // Again, y = betak P vk,  where  P = C**(-1).
    v = y;
    v *= (1.0/beta);

    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*beta0/
                              (rnorm*static_cast<double>(maxiter_)));
    mat_vec(v, y);
    if (i > 0) y.EqualsAXPlusBY(1.0, y, -beta/oldb, r1); 
    
    double alpha = InnerProd(v, y);
    y.EqualsAXPlusBY(1.0, y, -alpha/beta, r2);
    r1 = r2;
    r2 = y;    
    precond(r2, y);

    oldb = beta;
    beta = InnerProd(r2, y);
    if (beta < 0.0) {
      cerr << "MINRESSolver::Solve(): "
           << "matrix does not appear to be symmetric." << endl;
      throw(-1);
    }
    beta = sqrt(beta);
    tnorm2 = tnorm2 + alpha*alpha + oldb*oldb + beta*beta;

    if (i == 0) {
      // initialize a few things The following corresponds to y being the
      // solution of a generalized eigenvalue problem, Ay = lambda My.  This is
      // highly unlikely in our context, so we ignore this
      // if (beta/beta0 <= 10.0*kEpsilon) istop = -1; // beta2 = 0 or is ~ 0
      gmax = abs(alpha);
      gmin = gmax;
    }

    // Apply previous rotation Q_{k-1} to get
    //   [delta_k epsln_{k+1}] = [cs  sn][dbar_k    0       ]
    //   [gbar_k dbar_{k+1}]     [sn -cs][alpha_k beta_{k+1}].
    oldeps = epsln;
    double delta = cs*dbar + sn*alpha;
    double gbar  = sn*dbar - cs*alpha;
    epsln = sn*beta;
    dbar  = -cs*beta;

    // Compute the new plane rotation Q_k
    double gamma  = sqrt(gbar*gbar + beta*beta);
    cs     = gbar/gamma;
    sn     = beta/gamma;
    double phi    = cs*phibar;
    phibar = sn*phibar;
    
    // update solution
    double denom = 1.0/gamma;

    w1 = w2;
    w2 = w;
    w.EqualsAXPlusBY(denom, v, -oldeps*denom, w1);
    w.EqualsAXPlusBY(1.0, w, -delta*denom, w2);
    x.EqualsAXPlusBY(1.0, x, phi, w);
        
    gmax = std::max<double>(gmax, gamma);
    gmin = std::min<double>(gmin, gamma);
    double z = rhs1 /gamma;
    ynorm2 = z*z + ynorm2;
    rhs1 = rhs2 - delta*z;
    rhs2 = - epsln*z;

    // estimate various norms and test for convergence
    Anorm = sqrt(tnorm2);
    ynorm = sqrt(ynorm2);
    double epsx  = Anorm*ynorm*kEpsilon;

    qrnorm = phibar;
    rnorm = qrnorm;
    double rootl = sqrt(gbar*gbar + dbar*dbar);
    double relAnorm = rootl/Anorm;

    // Estimate  cond(A).
    // In this version we look at the diagonals of R in the factorization of the
    // lower Hessenberg matrix, Q * H = R, where H is the tridiagonal matrix
    // from Lanczos with one extra row, beta(k+1) e_k^T.
    Acond = gmax/gmin;

    // output relative residual norm and check stopping criteria
    writeKrylovHistory(fout, i+1, rnorm, beta0);
    iters++;
#if 0    
    if (Acond >= 0.1/kEpsilon) {
      fout << "MINRES stopping: estimate for cond(A) >= 0.1/eps." << endl;
      break;
    }
#endif
    if ( (rnorm <= epsx) || (relAnorm <= epsx) ) {
      fout << "MINRES stopping: residual has reached machine zero." << endl;
      break;
    }
    if ( rnorm <= beta0* ptin.get<double>("tol") ) break;
  }

  ptout.put<int>("iters", iters);
  ptout.put<double>("res", rnorm);
  
  if (ptin.get<bool>("check",false)) {
    // recalculate explicilty and check final residual
    mat_vec(x, w);
    // w = b - w = b - A*x
    w.EqualsAXPlusBY(1.0, b, -1.0, w);
    precond(w, y);
    double res = sqrt(InnerProd(w, y));
    ptout.put<double>("res", res);
    fout << "# MINRES final (true) residual : |res|/|res0| = "
         << res/beta0 << endl;    
    if (fabs(res - rnorm) > 0.01*ptin.get<double>("tol")*beta0) {
      fout << "# WARNING in MINRES: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - rnorm)/res0 = " << (res - rnorm)/beta0 << endl;
    }
  }
}
// ==============================================================================
template <class Vec>
FITRSolver<Vec>::FITRSolver() {
  maxiter_ = -1;
}
// ==============================================================================
template <class Vec>
void FITRSolver<Vec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "FITRSolver::SubspaceSize(): "
         << "illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  maxiter_ = m;

  // Note: STL vectors are used for V_ and Z_ while boost::numeric::ublas vectors
  // are used for double based vectors and matrices; this choice was made
  // mostly to help emphasize that the objects in V_ and Z_ are fundamentally
  // different (they are some sort of vectors themselves).
  Z_.reserve(maxiter_);
  V_.reserve(maxiter_+1);
  
  g_.resize(2*maxiter_+1, 0.0);
  y_.resize(maxiter_, 0.0);
  B_.resize(maxiter_+1, maxiter_, 0.0);
  H_.resize(maxiter_+1, maxiter_, 0.0);
}
// ==============================================================================
template <class Vec>
void FITRSolver<Vec>::MemoryRequired(ptree& num_required) const {
  if (maxiter_ < 1) {
    cerr << "FITRSolver::MemoryRequired(): "
         << "SubspaceSize must be called first (maxiter_ is undefined)" << endl;
    throw(-1);
  }
  num_required.put("num_vec", 2*maxiter_ + 1 /* Z and V*/
                   + 1 /* r_ */);
}
// ==============================================================================
template <class Vec>
void FITRSolver<Vec>::Solve(const ptree & ptin, const Vec & b, Vec & x,
                 MatrixVectorProduct<Vec> & mat_vec,
                 Preconditioner<Vec> & precond, ptree & ptout,
                 ostream & fout) {
  if (ptin.get<double>("radius") <= 0.0) {
    cerr << "FITRSolver::Solve(): "
         << "trust-region radius must be positive, radius = "
         << ptin.get<double>("radius") << endl;
    throw(-1);
  }
  Z_.clear();
  V_.clear();
  g_ = ublas::zero_vector<double>(g_.size());
  y_ = ublas::zero_vector<double>(y_.size());
  B_ = ublas::zero_matrix<double>(B_.size1(),B_.size2());
  H_ = ublas::zero_matrix<double>(H_.size1(),H_.size2());

  int iters = 0;

  // calculate the norm of the rhs vector and initialize the RHS of the reduced
  // system
  double norm0 = b.Norm2();
  V_.push_back(b);
  //V_[0] = b;
  double beta = norm0;
  g_(0) = beta;
  
  // normalize residual to get v_{0}
  r_.reset(new Vec);
  *r_ = b;
  V_[0] /= beta;
  int i = 0;
  
  // begin flexible Arnoldi iterations
  bool lin_depend = false;
  bool reset = false;
  double lambda = 0.0;
  double lambda_1 = 0.0;
  double lambda_2 = 0.0;
  bool active = false;
  double pred = 0.0;
  for (i = 0; i < maxiter_; i++) {
    iters++;

    // precondition the residual and store result in z[i]
#if 0
    precond.set_diagonal(std::max(
        lambda + (lambda - lambda_1) + (lambda - 2.0*lambda_1 + lambda_2), 0.0));
#endif
    //precond.set_diagonal(std::max(lambda + 2.0*(lambda - lambda_1), 0.0));
    precond.set_diagonal(std::max(lambda + 10.0*(lambda - lambda_1), 0.0));
    //precond.set_diagonal(std::max(lambda, 0.0));

#if 0
    if (i == 0)
      Z_[i] = *r_;
    else      
      precond(*r_, Z_[i]);
#endif

    // This was used for MOPTA 2013
    Z_.push_back(*r_);
    if (i > 0)
      precond(*r_, Z_[i]);    

#if 0
    if (i == 0)
      z[i] = v[i];
    else
      precond(v[i], z[i]);
#endif
    //cout << "FITRSolver::Solve(): after precondition..." << endl;
    
    // Orthonormalize Z_[i] against previous z vectors
    try {
      modGramSchmidt(i-1, Z_);
    } catch (string err_msg) {
      cerr << "FITRSolver::Solve(): "
           << "Z_ vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }
    
    //cout << "FITRSolver::Solve(): after GS Z..." << endl;
    
    // add to Krylov subspace
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
                              (beta*static_cast<double>(maxiter_)));
    V_.push_back(Z_[i]);
    mat_vec(Z_[i], V_[i+1]);
    
    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H_, V_);
    } catch (string err_msg) {
      cerr << "FITRSolver::Solve(): "
           << "V vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }
    //cout << "FITRSolver::Solve(): after GS V..." << endl;
    
    // add new row and column to matrix B_ = V_^{T} Z_
    for (int k = 0; k <= i; k++) {
      B_(k,i) = InnerProd(V_[k], Z_[i]);
      B_(i+1,k) = InnerProd(V_[i+1], Z_[k]);
    }
    //cout << "FITRSolver::Solve(): after adding rows/columns to B..." << endl;
    
    // compute the (symmetric part of) reduced matrix F = B^T H and rhs = B^T g,
    // then solve the reduced Trust region problem
    ublas::matrix<double> F(i+1, i+1, 0.0);
    ublas::vector<double> rhs(i+1);
    for (int j = 0; j < i+1; j++) {
      for (int k = 0; k < i+1; k++) {
        //F(j,k) = 0.0;
        for (int l = 0; l < i+2; l++) {
          double tmp = 0.5*B_(l,j)*H_(l,k);
          F(j,k) += tmp;
          F(k,j) += tmp;
        }
      }
      rhs(j) = -g_(0)*B_(0,j); // -ve sign is because b = -g
    }
    lambda_2 = lambda_1;
    lambda_1 = lambda;
    solveTrustReduced(i+1, F, ptin.get<double>("radius"), rhs, y_, lambda, pred);
    if (lambda > kEpsilon) active = true; 

#if 0
    // compute the residual norm
    ublas::matrix_range<ublas::matrix<double> >
        H_r(H, ublas::range(0,i+2), ublas::range(0,i+1)),
        B_r(B, ublas::range(0,i+2), ublas::range(0,i+1));
    beta = trustResidual(i+1, H_r, B_r, g, y, lambda);
#endif

    // compute the residual norm
    *r_ = b;
    for (int k = 0; k < i+1; k++) {
      // r = r - lambda*y[k]*z[k]
      r_->EqualsAXPlusBY(1.0, *r_, -lambda*y_(k), Z_[k]);
    }
    ublas::vector<double> Hy(i+2, 0.0);
    for (int k = 0; k < i+2; k++) {
      Hy(k) = 0.0;
      for (int j = 0; j < i+1; j++)
        Hy(k) += H_(k,j)*y_(j);
    }
    for (int k = 0; k < i+2; k++) {
      // r = r - V * H * y
      r_->EqualsAXPlusBY(1.0, *r_, -Hy(k), V_[k]);
    }    
    beta = r_->Norm2();

#if 0
    if ( (lambda > kEpsilon) && (!reset) ) {
      norm0 = beta;
      reset = true;
      cout << "new norm0 = " << norm0 << endl;
    }
#endif
    if (i == 0) {
      boost::format col_head("%|5| %|8t| %|-12| %|20t| %|-12|\n");
      col_head % "# iter" % "rel. res." % "lambda";
      writeKrylovHeader(fout, "FITR", ptin.get<double>("tol"), norm0, //beta,
                        col_head);
      //norm0 = beta;
    }
    fout << boost::format("%|5| %|8t| %|-12.6| %|20t| %|-12.6|\n")
        % (i+1) % (beta/norm0) % lambda;
    fout.flush();
    
    // check for convergence
    if ( (beta < ptin.get<double>("tol")*norm0) ||
         (beta < 1e-10) ) { //kEpsilon) ) { //1.e-12) ) {
      break;
    } else if (lin_depend) {
      cerr << "FITRSolver::Solve(): one of z[" << i << "] or  v[" << i+1 << "]"
           << " is linearly dependent, and residual is nonzero." << endl;
      cerr << "      ||res|| = " << beta << endl;
      throw(-1);
    }
  }

  // update the solution
  x = 0.0;
  for (int k = 0; k < std::min(i+1,maxiter_); k++) {
    // x = x + y[k]*z[k]
    x.EqualsAXPlusBY(1.0, x, y_(k), Z_[k]);
  }
  ptout.put<double>("pred", pred);
  ptout.put<int>("iters", iters);
  ptout.put<bool>("active", active);
  
  if (ptin.get<bool>("check", false)) {
    // check the residual norm
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(0.1*ptin.get<double>("tol")/
                              static_cast<double>(maxiter_));
    mat_vec(x, V_[0]);
    V_[0].EqualsAXPlusBY(1.0, V_[0], lambda, x);    
    // v[0] = b - v[0] = b - (lambdaI + A)*x
    V_[0].EqualsAXPlusBY(1.0, b, -1.0, V_[0]);
    double res = V_[0].Norm2();
    fout << "# FITR final (true) residual: |res|/|res0| = "
         << res/norm0 << endl;    
    if (fabs(res - beta) > 0.01*ptin.get<double>("tol")*norm0) {
      fout << "# WARNING in FITRSolver::Solve(): "
           << "true residual norm and calculated residual norm do not agree."
           << endl;
      fout << "# (res - beta)/res0 = " << (res - beta)/norm0 << endl;
    }
    // check that solution satisfies the trust region
    double x_norm = x.Norm2();
    if (fabs(lambda) < kEpsilon) {
      if (x_norm > ptin.get<double>("radius")) {
        cerr << "FITRSolver::Solve(): "
             << "lambda = 0 and solution is not inside trust region." << endl;
        throw(-1);
      }
    } else { // solution should lie on trust region
      if (x_norm - ptin.get<double>("radius")
          > 1.e-4*ptin.get<double>("radius")) {
        cerr << "FITRSolver::Solve(): lambda > 0 and solution is not on trust "
             << "region." << endl;
        cerr << "x_norm - radius = " << x_norm - ptin.get<double>("radius")
             << endl;     
        //throw(-1);
      }
    }
  }
}
// ==============================================================================
#if 0
template <class Vec>
void FITRSolver<Vec>::Solve(const ptree & ptin, const Vec & b, Vec & x,
                 MatrixVectorProduct<Vec> & mat_vec,
                 Preconditioner<Vec> & precond, ptree & ptout,
                 ostream & fout) {
  if (ptin.get<double>("radius") <= 0.0) {
    cerr << "FITRSolver::Solve(): "
         << "trust-region radius must be positive, radius = "
         << ptin.get<double>("radius") << endl;
    throw(-1);
  }
  g_ = ublas::zero_vector<double>(g_.size());
  //g_.clear();
  y_.clear();
  B_.clear();
  H_.clear();
  int iters = 0;

  // calculate the norm of the rhs vector and initialize the RHS of the reduced
  // system
  double norm0 = b.Norm2();
  V_.push_back(b);
  //V_[0] = b;
  double beta = norm0;
  g_(0) = beta;
  
  // normalize residual to get v_{0} 
  r_ = b;
  V_[0] /= beta;
  int i = 0;
  
  // begin flexible Arnoldi iterations
  bool lin_depend = false;
  bool reset = false;
  double lambda = 0.0;
  double lambda_1 = 0.0;
  double lambda_2 = 0.0;
  bool active = false;
  double pred = 0.0;
  for (i = 0; i < maxiter_; i++) {
    iters++;

    // precondition the residual and store result in z[i]
#if 0
    precond.set_diagonal(std::max(
        lambda + (lambda - lambda_1) + (lambda - 2.0*lambda_1 + lambda_2), 0.0));
#endif
    //precond.set_diagonal(std::max(lambda + 2.0*(lambda - lambda_1), 0.0));
    precond.set_diagonal(std::max(lambda + 10.0*(lambda - lambda_1), 0.0));
    //precond.set_diagonal(std::max(lambda, 0.0));

#if 1
    Z_.push_back(r_);
    if (i > 0)
      precond(r_, Z_[i]);
#else
    Z_.push_back(V_[i]);
    if (i > 0)
      precond(V_[i], Z_[i]);
#endif

    //cout << "FITRSolver::Solve(): after precondition..." << endl;

    // Orthonormalize Z_[i] against previous z vectors
    try {
      modGramSchmidt(i-1, Z_);
    } catch (string err_msg) {
      cerr << "FITRSolver::Solve(): "
           << "Z_ vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }
    
    //cout << "FITRSolver::Solve(): after GS Z..." << endl;
    
    // add to Krylov subspace
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
                              (beta*static_cast<double>(maxiter_)));
    V_.push_back(Z_[i]);
    mat_vec(Z_[i], V_[i+1]);
    
    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H_, V_);
    } catch (string err_msg) {
      cerr << "FITRSolver::Solve(): "
           << "V vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }
    //cout << "FITRSolver::Solve(): after GS V..." << endl;
    
    // add new row and column to matrix B_ = V_^{T} Z_
    for (int k = 0; k <= i; k++) {
      B_(k,i) = InnerProd(V_[k], Z_[i]);
      B_(i+1,k) = InnerProd(V_[i+1], Z_[k]);
    }
    //cout << "FITRSolver::Solve(): after adding rows/columns to B..." << endl;
    
    // compute the (symmetric part of) reduced matrix F = B^T H and rhs = B^T g,
    // then solve the reduced Trust region problem
    ublas::matrix<double> F(i+1, i+1, 0.0);
    ublas::vector<double> rhs(i+1);
    for (int j = 0; j < i+1; j++) {
      for (int k = 0; k < i+1; k++) {
        //F(j,k) = 0.0;
        for (int l = 0; l < i+2; l++) {
          double tmp = 0.5*B_(l,j)*H_(l,k);
          F(j,k) += tmp;
          F(k,j) += tmp;
        }
      }
      rhs(j) = -g_(0)*B_(0,j); // -ve sign is because b = -g
    }
    lambda_2 = lambda_1;
    lambda_1 = lambda;
    solveTrustReduced(i+1, F, ptin.get<double>("radius"), rhs, y_, lambda, pred);
    if (lambda > kEpsilon) active = true; 

    // project (lambda + A)Zy - b = lambda Zy + VHy onto V, and solve for y
    ublas::matrix_range<ublas::matrix<double> >
        H_r(H_, ublas::range(0,i+1), ublas::range(0,i+1)),
        B_r(B_, ublas::range(0,i+1), ublas::range(0,i+1));
    F = lambda*B_r + H_r;
    solveReduced(i+1, F, g_, y_);
    
#if 0
    // compute the residual norm
    ublas::matrix_range<ublas::matrix<double> >
        H_r(H, ublas::range(0,i+2), ublas::range(0,i+1)),
        B_r(B, ublas::range(0,i+2), ublas::range(0,i+1));
    beta = trustResidual(i+1, H_r, B_r, g, y, lambda);
#endif

    // compute the residual norm
    r_ = b;
    for (int k = 0; k < i+1; k++) {
      // r = r - lambda*y[k]*z[k]
      r_.EqualsAXPlusBY(1.0, r_, -lambda*y_(k), Z_[k]);
    }
    ublas::vector<double> Hy(i+2, 0.0);
    for (int k = 0; k < i+2; k++) {
      Hy(k) = 0.0;
      for (int j = 0; j < i+1; j++)
        Hy(k) += H_(k,j)*y_(j);
    }
    for (int k = 0; k < i+2; k++) {
      // r = r - V * H * y
      r_.EqualsAXPlusBY(1.0, r_, -Hy(k), V_[k]);
    }    
    beta = r_.Norm2();

#if 0
    if ( (lambda > kEpsilon) && (!reset) ) {
      norm0 = beta;
      reset = true;
      cout << "new norm0 = " << norm0 << endl;
    }
#endif
    if (i == 0) {
      boost::format col_head("%|5| %|8t| %|-12| %|20t| %|-12|\n");
      col_head % "# iter" % "rel. res." % "lambda";
      writeKrylovHeader(fout, "FITR", ptin.get<double>("tol"), beta,
                        col_head);
      norm0 = beta;
    }
    fout << boost::format("%|5| %|8t| %|-12.6| %|20t| %|-12.6|\n")
        % (i+1) % (beta/norm0) % lambda;
    fout.flush();
    
    // check for convergence
    if ( (beta < ptin.get<double>("tol")*norm0) ||
         (beta < kEpsilon) ) { //1.e-12) ) {
      break;
    } else if (lin_depend) {
      cerr << "FITRSolver::Solve(): one of z[" << i << "] or  v[" << i+1 << "]"
           << " is linearly dependent, and residual is nonzero." << endl;
      cerr << "      ||res|| = " << beta << endl;
      throw(-1);
    }
  }

  // update the solution
  x = 0.0;
  for (int k = 0; k < std::min(i+1,maxiter_); k++) {
    // x = x + y[k]*z[k]
    x.EqualsAXPlusBY(1.0, x, y_(k), Z_[k]);
  }
  ptout.put<double>("pred", pred);
  ptout.put<int>("iters", iters);
  ptout.put<bool>("active", active);
  
  if (ptin.get<bool>("check", false)) {
    // check the residual norm
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(0.1*ptin.get<double>("tol")/
                              static_cast<double>(maxiter_));
    mat_vec(x, V_[0]);
    V_[0].EqualsAXPlusBY(1.0, V_[0], lambda, x);    
    // v[0] = b - v[0] = b - (lambdaI + A)*x
    V_[0].EqualsAXPlusBY(1.0, b, -1.0, V_[0]);
    double res = V_[0].Norm2();
    fout << "# FITR final (true) residual: |res|/|res0| = "
         << res/norm0 << endl;    
    if (fabs(res - beta) > 0.01*ptin.get<double>("tol")*norm0) {
      fout << "# WARNING in FITRSolver::Solve(): "
           << "true residual norm and calculated residual norm do not agree."
           << endl;
      fout << "# (res - beta)/res0 = " << (res - beta)/norm0 << endl;
    }
    // check that solution satisfies the trust region
    double x_norm = x.Norm2();
    if (fabs(lambda) < kEpsilon) {
      if (x_norm > ptin.get<double>("radius")) {
        cerr << "FITRSolver::Solve(): "
             << "lambda = 0 and solution is not inside trust region." << endl;
        throw(-1);
      }
    } else { // solution should lie on trust region
      if (x_norm - ptin.get<double>("radius")
          > 1.e-6*ptin.get<double>("radius")) {
        cerr << "FITRSolver::Solve(): lambda > 0 and solution is not on trust "
             << "region." << endl;
        cerr << "x_norm - radius = " << x_norm - ptin.get<double>("radius")
             << endl;     
        throw(-1);
      }
    }
  }
}
#endif
// ==============================================================================
template <class Vec>
void FITRSolver<Vec>::ReSolve(const ptree & ptin, const Vec & b, Vec & x,
                        ptree & ptout, ostream & fout) {
  if (ptin.get<double>("radius") <= 0.0) {
    cerr << "FITRSolver::ReSolve(): "
         << "trust-region radius must be positive, radius = "
         << ptin.get<double>("radius") << endl;
    throw(-1);
  }

  int i = Z_.size()-1;
  // compute the (symmetric part of) reduced matrix F = B^T H and rhs = B^T g,
  // then resolve the reduced Trust region problem
  ublas::matrix<double> F(i+1, i+1, 0.0);
  ublas::vector<double> rhs(i+1);
  for (int j = 0; j < i+1; j++) {
    for (int k = 0; k < i+1; k++) {
      //F(j,k) = 0.0;
      for (int l = 0; l < i+2; l++) {
        double tmp = 0.5*B_(l,j)*H_(l,k);
        F(j,k) += tmp;
        F(k,j) += tmp;
      }
    }
    rhs(j) = -g_(0)*B_(0,j); // -ve sign is because b = -g
  }
  double lambda = 0.0;
  double pred = 0.0;
  bool active = false;
  solveTrustReduced(i+1, F, ptin.get<double>("radius"), rhs, y_, lambda, pred);
  if (lambda > kEpsilon) active = true;

  // compute the residual norm
  r_.reset(new Vec);
  *r_ = b;
  for (int k = 0; k < i+1; k++) {
    // r = r - lambda*y[k]*z[k]
    r_->EqualsAXPlusBY(1.0, *r_, -lambda*y_(k), Z_[k]);
  }
  ublas::vector<double> Hy(i+2, 0.0);
  for (int k = 0; k < i+2; k++) {
    Hy(k) = 0.0;
    for (int j = 0; j < i+1; j++)
      Hy(k) += H_(k,j)*y_(j);
  }
  for (int k = 0; k < i+2; k++) {
    // r = r - V * H * y
    r_->EqualsAXPlusBY(1.0, *r_, -Hy(k), V_[k]);
  }    
  double beta = r_->Norm2();
  double norm0 = b.Norm2();

  fout << "# FITRSolver: resolving at radius = "
       << ptin.get<double>("radius") << endl;
  fout << boost::format("%|5| %|8t| %|-12.6| %|20t| %|-12.6|\n")
      % (i+1) % (beta/norm0) % lambda;
  fout.flush();
  
  // update the solution
  x = 0.0;
  for (int k = 0; k < std::min(i+1,maxiter_); k++) {
    // x = x + y[k]*z[k]
    x.EqualsAXPlusBY(1.0, x, y_(k), Z_[k]);
  }
  ptout.put<double>("pred", pred);
  cout << "ptout.get(pred) = " << ptout.get<double>("pred") << endl;
  ptout.put<int>("iters", 0);
  ptout.put<bool>("active", active);
}
// ==============================================================================
template <class Vec>
STCGSolver<Vec>::STCGSolver() {
  maxiter_ = -1;
}
// ==============================================================================
template <class Vec>
void STCGSolver<Vec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "STCGSolver::SubspaceSize(): "
         << "illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  maxiter_ = m;
  work_.reserve(7);
}
// ==============================================================================
template <class Vec>
void STCGSolver<Vec>::MemoryRequired(ptree& num_required) const {
  num_required.put("num_vec", 4);
}
// ==============================================================================
template <class Vec>
void STCGSolver<Vec>::Solve(const ptree & ptin, const Vec & b, Vec & x,
                              MatrixVectorProduct<Vec,Vec> & mat_vec,
                              Preconditioner<Vec,Vec> & precond,
                              ptree & ptout, ostream & fout) {
  if (ptin.get<double>("radius") <= 0.0) {
    cerr << "STCGSolver::Solve(): "
         << "trust-region radius must be positive, radius = "
         << ptin.get<double>("radius") << endl;
    throw(-1);
  }    
  // allocate memory, and set some references for readability
  work_.resize(4, x);
  Vec& r = work_[0];
  Vec& z = work_[1]; 
  Vec& p = work_[2];
  Vec& Ap = work_[3];

  // if using projected CG, the termination criterion is different
  bool proj_cg = ptin.get<bool>("proj_cg", false);
  
  // define various vectors and scalars
  r = b;
  x = 0.0;
  double alpha;  
  double x_norm2 = 0.0;
  int iters = 0;

  // calculate the norm of the rhs vector
  double norm0 = r.Norm2();
  double res_norm2 = norm0;
  precond(r, z);
  double rdotz = InnerProd(r, z);
  if (proj_cg) norm0 = rdotz;

  writeKrylovHeader(fout, "STCG", ptin.get<double>("tol"), norm0);
  writeKrylovHistory(fout, iters, norm0, norm0);
          
  // check if solved already
#if 0
  // absolute tolerance needs to be an input parameter
  if (norm0 < kEpsilon) {
    // system is already solved
    fout << "# STCG: system solved by initial guess." << endl;
    ptout.put<double>("pred", 0.0);
    ptout.put<int>("iters", iters);
    ptout.put<double>("res", 0.0);
    ptout.put<bool>("active", false);
    return;
  }
#endif
 
  p = z;
  Ap = p;

  // loop over search directions
  double radius = ptin.get<double>("radius");
  bool active = false;
  int i = 0;
  for (i = 0; i < maxiter_; i++) {
    iters++;

    if (ptin.get<bool>("dynamic",false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
                              (res_norm2*static_cast<double>(maxiter_)));
    mat_vec(p, Ap);
    alpha = InnerProd(p, Ap);
    if (alpha <= -1e-8) { // WARNING: hard-coded number here
      // direction of nonpositive curvature detected
      double xp = InnerProd(x, p);
      double x2 = x_norm2*x_norm2;
      double p2 = InnerProd(p, p);
      if (p2 > kEpsilon) {
        double tau = (-xp + sqrt(xp*xp - p2*(x2 - radius*radius)))/p2;
        x.EqualsAXPlusBY(1.0, x, tau, p);
        r.EqualsAXPlusBY(1.0, r, -tau, Ap);
        if (proj_cg) {
          precond(r, z);
          rdotz = InnerProd(r, z);
        }
      }
      res_norm2 = r.Norm2();
      // if (iters == 1) {
      //   writeKrylovHeader(fout, "STCG", ptin.get<double>("tol"),
      //                     res_norm2);
      //   norm0 = res_norm2;
      // }
      if (proj_cg)
        writeKrylovHistory(fout, iters, rdotz, norm0);
      else
        writeKrylovHistory(fout, iters, res_norm2, norm0);
      fout << "# direction of nonpositive curvature detected: ";
      fout << "alpha = " << alpha << endl;
      active = true;
      break;
    }

    alpha = rdotz/alpha;
    x.EqualsAXPlusBY(1.0, x, alpha, p);
    x_norm2 = x.Norm2();
    if (x_norm2 >= radius) {
      x.EqualsAXPlusBY(1.0, x, -alpha, p);
      double xp = InnerProd(x, p);
      double x2 = InnerProd(x, x);
      double p2 = InnerProd(p, p);
      double tau = (-xp + sqrt(xp*xp - p2*(x2 - radius*radius)))/p2;
      x.EqualsAXPlusBY(1.0, x, tau, p);
      r.EqualsAXPlusBY(1.0, r, -tau, Ap);      
      res_norm2 = r.Norm2();
      // if (iters == 1) {      
      //   writeKrylovHeader(fout, "STCG", ptin.get<double>("tol"),
      //                     res_norm2);
      //   norm0 = res_norm2;
      // }
      if (proj_cg) {
        precond(r, z);
        rdotz = InnerProd(r, z);
        writeKrylovHistory(fout, iters, rdotz, norm0);
      } else
        writeKrylovHistory(fout, iters, res_norm2, norm0);      
      fout << "# trust-region boundary encountered" << endl;
      active = true;
      break;
    }

    // compute residual
    r.EqualsAXPlusBY(1.0, r, -alpha, Ap);
    res_norm2 = r.Norm2();
    precond(r, z);
    double beta = 1.0/rdotz;
    rdotz = InnerProd(r, z);

    // check for convergence
    if (proj_cg) {
      writeKrylovHistory(fout, iters, rdotz, norm0);
      if (rdotz < norm0*ptin.get<double>("tol"))
        break;
    } else {
      writeKrylovHistory(fout, iters, res_norm2, norm0);
      if (res_norm2 < norm0*ptin.get<double>("tol"))
        break;
    }

    beta *= rdotz;
    p.EqualsAXPlusBY(1.0, z, beta, p);    
  }

  // compute the predicted reduction in the objective, and store output data
  // Note: -Ax = r - b --> x^T b - 0.5 x^T A x = 0.5*x^T(b + r)
  r.EqualsAXPlusBY(1.0, r, 1.0, b);
  ptout.put<double>("pred", 0.5*InnerProd(x, r));
  ptout.put<int>("iters", iters);
  if (proj_cg)
    ptout.put<double>("res", rdotz);
  else
    ptout.put<double>("res", res_norm2);
  ptout.put<bool>("active", active);
  
  if (ptin.get<bool>("check", false)) {
    // check the residual norm
    if (ptin.get<bool>("dynamic",false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*0.1/
                              static_cast<double>(maxiter_));
    mat_vec(x, r);
    r.EqualsAXPlusBY(1.0, b, -1.0, r);
    double res;
    if (proj_cg) {
      precond(r, z);
      res = InnerProd(r, z);
      fout << "# STCG final (true) residual : |res|/|res0| = "
           << res/norm0 << endl;
      if (fabs(res - rdotz) > 0.01*ptin.get<double>("tol")*norm0) {
        fout << "# WARNING in STCGSolver::Solver(): "
             << "true residual norm and calculated residual norm do not agree."
             << endl;
        fout << "# (res - beta)/res0 = " << (res - rdotz)/norm0 << endl;
      }      
    } else { // not using projected CG
      res = r.Norm2();
      fout << "# STCG final (true) residual : |res|/|res0| = "
           << res/norm0 << endl;
      if (fabs(res - res_norm2) > 0.01*ptin.get<double>("tol")*norm0) {
        fout << "# WARNING in STCGSolver::Solver(): "
             << "true residual norm and calculated residual norm do not agree."
             << endl;
        fout << "# (res - beta)/res0 = " << (res - res_norm2)/norm0 << endl;
      }
    }
    // check that solution satisfies the trust region
    x_norm2 = x.Norm2();
    if (x_norm2 - radius > 1e-6) {
      cerr << "STCGSolver::Solver(): solution is outside trust region."
           << endl;
      throw(-1);
    }
  }  
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
FFOMWithSMART<Vec,PrimVec,DualVec>::FFOMWithSMART() {
  maxiter_ = -1;
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FFOMWithSMART<Vec,PrimVec,DualVec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "FFOMWithSMART::SubspaceSize(): "
         << "illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  maxiter_ = m;

  // Note: STL vectors are used for V_ and Z_ while boost::numeric::ublas vectors
  // are used for double based vectors and matrices; this choice was made
  // mostly to help emphasize that the objects in V_ and Z_ are fundamentally
  // different (they are some sort of vectors themselves).
  V_.reserve(maxiter_+1);
  Z_.reserve(maxiter_);
  g_.resize(maxiter_+1, 0.0);
  sn_.resize(maxiter_+1, 0.0);
  cs_.resize(maxiter_+1, 0.0);
  y_.resize(maxiter_, 0.0);
  B_.resize(maxiter_+1, maxiter_, 0.0);
  VtZ_.resize(maxiter_+1, maxiter_, 0.0);
  VtZprim_.resize(maxiter_+1, maxiter_, 0.0);
  VtZdual_.resize(maxiter_+1, maxiter_, 0.0);
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FFOMWithSMART<Vec,PrimVec,DualVec>::MemoryRequired(
    ptree& num_required) const {
  if (maxiter_ < 1) {
    cerr << "FFOMWithSMART::MemoryRequired(): "
         << "SubspaceSize must be called first (maxiter_ is undefined)" << endl;
    throw(-1);
  }
  num_required.put("num_vec", maxiter_+1 /* for V_ */ + maxiter_ /* for Z_ */
                   + 1 /* for res_ */ );
  num_required.put("num_primal", 1 /* for neg_grad */);
  num_required.put("num_dual", 1 /* for Ap_ */);
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FFOMWithSMART<Vec,PrimVec,DualVec>::Solve(
    const ptree & ptin, const Vec & b, Vec & x, 
    MatrixVectorProduct<Vec,Vec> & mat_vec, Preconditioner<Vec,Vec> & precond,
    ptree & ptout, ostream & fout) {
  V_.clear();
  Z_.clear();
  g_ = ublas::zero_vector<double>(g_.size());
  //gu_ = ublas::zero_vector<double>(gu_.size());
  ublas::vector<double> gu(g_.size(), 0.0);
  sn_ = ublas::zero_vector<double>(sn_.size());
  cs_ = ublas::zero_vector<double>(cs_.size());
  y_ = ublas::zero_vector<double>(y_.size());
  B_ = ublas::zero_matrix<double>(B_.size1(),B_.size2());
  VtZ_ = ublas::zero_matrix<double>(VtZ_.size1(),VtZ_.size2());
  VtZprim_ = VtZ_;
  VtZdual_ = VtZ_;
  // TODO: consider making res_ and Ap_ objects, if no ReSolve method is needed
  res_.reset(new Vec);
  Ap_.reset(new DualVec);
  PrimVec neg_grad(x.primal());
  neg_grad *= -1.0;
  int iters = 0;
  cout << "After initializing vectors..." << endl;
  
  // calculate the norm of the rhs vector
  double norm0 = b.Norm2();
  double primal_norm0 = b.primal().Norm2();
  double dual_norm0 = b.dual().Norm2();

  // calculate the initial residual (actually the negative residual)
  // and compute its norm
  if (ptin.get<bool>("dynamic", false))
    mat_vec.set_product_tol(ptin.get<double>("tol")/
                            static_cast<double>(maxiter_));
  V_.push_back(b); // needed to initialize W_.[0], unfortunately
  V_[0] *= -1.0;
  //mat_vec(x, V_[0]);
  //V_[0] -= b;
  
  double beta = V_[0].Norm2();
#if 0
  if ( (beta < ptin.get<double>("tol")*norm0) || (beta < kEpsilon) ) {
    // system is already solved
    fout << "FFOM system solved by initial guess." << endl;
    ptout.put<double>("res", beta);
    return;
  }
#endif

  // normalize residual to get w_{0} (the negative sign is because W_[0]
  // holds the negative residual, as mentioned above)
  V_[0] /= -beta;

  // initialize the RHS of the reduced system
  g_(0) = beta;
  gu(0) = InnerProd(V_[0].primal(), b.primal());

  // output header information including initial residual
  int i = 0;
#if 0
  writeKrylovHeader(fout, "FFOM with SMART Tests",
                    ptin.get<double>("tol"), beta);
  writeKrylovHistory(fout, i, beta, norm0);
#endif

  cout << "Just before loop..." << endl;
  
  bool lin_depend = false;
  double primal_norm = primal_norm0;
  double dual_norm = dual_norm0;
  double grad_dot_search = 0.0; 
  double pi = ptin.get<double>("pi");
  double mu = 0.0;
  // loop over all search directions
  for (i = 0; i < maxiter_; i++) {
    // check if solution has converged
    //if (beta < ptin.get<double>("tol")*norm0) break;
    iters++;

    // precondition the Vec V_[i] and store result in Z_[i]
    Z_.push_back(b);
    precond(V_[i], Z_[i]);
    
    // add to Krylov subspace
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
                              (beta*static_cast<double>(maxiter_)));
    V_.push_back(Z_[i]);
    mat_vec(Z_[i], V_[i+1]);

    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, B_, V_);
    } catch (string err_msg) {
      cerr << "FFOM: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

#if 0
    // apply old Givens rotations to new column of the Hessenberg matrix
    for (int k = 0; k < i; k++)
      applyGivens(sn_(k), cs_(k), B_(k,i), B_(k+1,i));
#endif

    cout << "After preconditioning, matvec, and Gram-Schmidt..." << endl;
    
    // form the new row and column of V^T Z and its primal and dual parts
    for (int k = 0; k <= i; k++) {
      VtZprim_(k, i) = InnerProd(V_[k].primal(), Z_[i].primal());
      VtZprim_(i+1, k) = InnerProd(V_[i+1].primal(), Z_[k].primal());
      VtZdual_(k, i) = InnerProd(V_[k].dual(), Z_[i].dual());
      VtZdual_(i+1, k) = InnerProd(V_[i+1].dual(), Z_[k].dual());
      VtZ_(k, i) = VtZprim_(k, i) + VtZdual_(k, i);
      VtZ_(i+1, k) = VtZprim_(i+1, k) + VtZdual_(i+1, k);
    }

    // form the new row in gu
    gu(i+1) = InnerProd(V_[i+1].primal(), b.primal());
    
    // The following vector and matrix ranges permit easier manipulation
    ublas::vector_range<ublas::vector<double> >
        y_r(y_, ublas::range(0,i+1)),
        g_r(g_, ublas::range(0,i+1));
    ublas::matrix_range<ublas::matrix<double> >
        B_r(B_, ublas::range(0,i+2), ublas::range(0,i+1)),
        VtZ_r(VtZ_, ublas::range(0,i+2), ublas::range(0,i+1)),
        VtZprim_r(VtZprim_, ublas::range(0,i+2), ublas::range(0,i+1)),
        VtZdual_r(VtZdual_, ublas::range(0,i+2), ublas::range(0,i+1));    
    
    // compute the current solution, which is needed for the SMART tests, as
    // well as upsilon and nu
    ublas::matrix<double> A = mu*VtZprim_r + B_r;
    double upsilon = 0.0, nu = 0.0;
    ComputeSMARTUpsilon(i+1, Z_, A, gu, x.primal(), upsilon);
    // A is (i+2)x(i+1), but solveReducedMultipleRHS takes only square part
    solveReduced(i+1, A, g_, y_);
    x = 0.0;
    for (int k = 0; k <= i; k++) // x = x + y[k]*z[k]
      x.EqualsAXPlusBY(1.0, x, y_[k], Z_[k]);
    nu = std::max(0.0, InnerProd(x.primal(), x.primal()) - upsilon);

    cout << "After solving reduced system and setting solution..." << endl;
    cout << "upsilon = " << upsilon << ": nu = " << nu << endl;
    
    // compute Ap and residual, which are also required by the SMART tests
    // NOTE: ||r_|| can be found using the reduced solution, but only when mu=0
    ublas::vector<double> By = ublas::prod(B_r, y_r);
    *res_ = 0.0;
    for (int k = 0; k <= i+1; k++) // res += V*B*y
      res_->EqualsAXPlusBY(1.0, *res_, By(k), V_[k]);
    *Ap_ = res_->dual();
    res_->EqualsAXPlusBY(1.0, b, -1.0, *res_); // b.primal() = -grad
    // add diagonal term to residual, if necessary
    if (mu > 0.0) {
      for (int k = 0; k <= i; k++)
        res_->primal().EqualsAXPlusBY(1.0, res_->primal(),
                                      -mu*y_[k], Z_[k].primal());
    }
    beta = res_->Norm2();
    primal_norm = res_->primal().Norm2();
    dual_norm = res_->dual().Norm2();

    cout << "After computing residual and Ap..." << endl;
    
    // check the SMART tests 1 and 2
    double A_norm = ptin.get<double>("A_norm_estimate"); // an input for now...
#if 0
    double upsilon = 0.0, nu = 0.0;
    ComputeSMARTUpsilonAndNu(x.primal(), *Ap_, A_norm, upsilon, nu);
#endif
    cout << "Before pHp..." << endl;
    double pHp =
        ublas::inner_prod(y_r, ublas::prod(ublas::trans(VtZprim_r), By))
        - 2.0*InnerProd(x.dual(), *Ap_)
        + mu*InnerProd(x.primal(), x.primal());
    cout << "after pHp..." << endl;
    double sigma = ptin.get<double>("tau", 0.2)
        *(1.0 - ptin.get<double>("dual_tol", 0.01));
    grad_dot_search = -InnerProd(neg_grad, x.primal());
    bool model_red = CheckSMARTModelReduction(
        grad_dot_search, pHp, ptin.get<double>("theta"), upsilon, sigma, pi,
        dual_norm0, dual_norm);

    cout << "After checking model reduction..." << endl;
    
    // output to history file here
    if (i == 0) {
      boost::format col_head(string("%|6| %|8t| %|-12| %|20t| %|-12|")+
                             string(" %|32t| %|-12| %|44t| %|-12|")+
                             string(" %|56t| %|-12|\n"));
      col_head % "# iter" % "rel. res." % "rel. grad." % "rel. feas." % "pred"
          % "mu";
      writeKrylovHeader(fout, "FFOMWithSMART", ptin.get<double>("tol"),
                        norm0, col_head);
    }
    double pred = -grad_dot_search + pi*(dual_norm0 - dual_norm);
    fout << boost::format(string("%|6| %|8t| %|-12.6| %|20t| %|-12.6|")+
                          string(" %|32t| %|-12.6| %|44t| %|-12.6|")+
                          string(" %|56t| %|-12.6|\n"))
        % (i+1) % (beta/norm0) % (primal_norm/primal_norm0)
        % (dual_norm/dual_norm0) % pred % mu;
    fout.flush();
    
    // check Termination condition 1
    if ( (model_red) && (beta <= ptin.get<double>("tol")*norm0) ) {
      fout << "# Termination condition 1 satisfied (rel res norm < tol)" << endl;
      break;
    }

    // check Termination condition 2
    if ( (primal_norm <= ptin.get<double>("primal_tol", 10.0)*primal_norm0) &&
         (dual_norm <= ptin.get<double>("dual_tol", 0.01)*dual_norm0) ) {
      if ( (0.5*pHp >= ptin.get<double>("theta")*upsilon) ||
           (ptin.get<double>("psi", 10.0)*nu >= upsilon) ) {
        // condition 2 is satisfied; check to see if penalty needs adjustment
        fout << "# Termination condition 2 satisfied (constraint reduction)\n";
        if (!model_red) {
          pi = UpdatePenaltyParameter(
              grad_dot_search, pHp, ptin.get<double>("theta"), upsilon,
              ptin.get<double>("tau", 0.2), dual_norm0, dual_norm);
          fout << "# Penalty parameter updated: pi_new = " << pi << endl;
        }
        break;
      }
    }

    cout << "After checking test 1 and test 2..." << endl;
    cout << "pHp = " << pHp << endl;
    
    // If we get here, we failed to converge; check if the Hessian should be
    // modified or not
    if ( (!model_red) && (0.5*pHp < ptin.get<double>("theta")*upsilon) &&
         (ptin.get<double>("psi", 10.0)*nu < upsilon) && (i != maxiter_-1) ) {
      // Construct the reduced Hessian
      ublas::matrix<double> Ared = ublas::prod(ublas::trans(VtZdual_r),B_r);
      ublas::matrix<double> Hred = ublas::prod(ublas::trans(VtZ_r),B_r)
          - Ared - ublas::trans(Ared);
      cout << "Hred:" << endl;
      for (int j = 0; j < i+1; j++) {
        for (int k = 0; k < i+1; k++) 
          cout << Hred(j,k) << " ";
        cout << endl;
      }
      ublas::vector<double> eig(i+1,0.0);
      ublas::matrix<double> E(i+1,i+1,0.0);
      eigenvaluesAndVectors(i+1, Hred, eig, E);
      ublas::vector<double> u = ublas::prod(ublas::trans(E), y_r);
      double u2 = ublas::inner_prod(u, u);
      double u2eig = 0.0;
      for (int k = 0; k < i+1; k++) u2eig += u(k)*eig(k)*u(k);
      cout << "Hred eigenvalues = ";
      for (int k = 0; k < i+1; k++) cout << eig(k) << " ";
      cout << endl;
      cout << "mu estimate = " << -u2eig/u2 << endl;
      mu = std::max(0.0, -2.0*u2eig/u2);
      //mu = std::max(10.0*mu, ptin.get<double>("mu_init", 1e-4));
    }
    
#if 0
    // Generate the new Givens rotation matrix and apply it to
    // the last two elements of B(i,:) and g
    generateGivens(B_(i,i), B_(i+1,i), sn_(i), cs_(i));
    applyGivens(sn_(i), cs_(i), g_(i), g_(i+1));
#endif
    
    cout << "end of loop..." << endl;
  }
  i = iters;
  
  ptout.put("iters", iters);
  ptout.put("res", beta);
  ptout.put("pi", pi);
  ptout.put("model_reduction", -grad_dot_search + pi*(dual_norm0 - dual_norm));
  ptout.put("mu", mu);
  
  if (ptin.get<bool>("check", false)) {
    // recalculate explicilty and check final residual
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(0.1*ptin.get<double>("tol")/
                              static_cast<double>(maxiter_));    
    mat_vec(x, *res_);
    // res_ = b - W_[0] = b - A*x
    res_->EqualsAXPlusBY(1.0, b, -1.0, *res_);
    // add diagonal term to residual, if necessary
    if (mu > 0.0)
      for (int k = 0; k < i; k++)
        res_->primal().EqualsAXPlusBY(1.0, res_->primal(),
                                      -mu*y_[k], Z_[k].primal());
    double true_res = res_->Norm2();
    fout << "# FFOMWithSMART final (true) residual : |res|/|res0| = "
         << true_res/norm0 << endl;
    ptout.put<double>("res", true_res);
    if (fabs(true_res - beta) > 0.01*ptin.get<double>("tol")*norm0) {
      fout << "# WARNING in FFOMWithSMART: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = " << (true_res - beta)/norm0 << endl;
      fout << "# rel. primal res norm = " << res_->primal().Norm2()/primal_norm0
           << endl;
      fout << "# rel. dual res norm   = " << res_->dual().Norm2()/dual_norm0
           << endl;
    }
  }
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FFOMWithSMART<Vec,PrimVec,DualVec>::ComputeSMARTUpsilonAndNu(
    const PrimVec& p, const DualVec& Ap, const double& A_norm, double& upsilon,
    double& nu) const {
  double p_norm2 = InnerProd(p,p);
  double Ap_norm2 = InnerProd(Ap, Ap);
  nu = Ap_norm2/A_norm;
  upsilon = p_norm2 - nu;
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FFOMWithSMART<Vec,PrimVec,DualVec>::ComputeSMARTUpsilon(
    int i, const std::vector<Vec>& Z, const ublas::matrix<double>& A,
    const ublas::vector<double>& gu, PrimVec& work, double& upsilon) const {
  ublas::vector<double> y(i, 0.0);
  solveReduced(i, A, gu, y);
  work = 0.0;
  for (int k = 0; k < i; k++) // x = x + y[k]*z[k]
    work.EqualsAXPlusBY(1.0, work, y(k), Z_[k].primal());
  upsilon = InnerProd(work, work);
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
bool FFOMWithSMART<Vec,PrimVec,DualVec>::CheckSMARTModelReduction(
    const double& grad_dot_search, const double& pHp,
    const double& theta, const double& upsilon, const double& sigma,
    const double& pi, const double& dual_norm0, const double& dual_norm) const {
  double model_reduction = -grad_dot_search + pi*(dual_norm0 - dual_norm);
  double rhs = std::max(0.5*pHp, theta*upsilon)
      + sigma*pi*std::max(dual_norm0, dual_norm - dual_norm0);
  cout << "model_reduction = " << model_reduction
       << ": rhs_part_1 = " << std::max(0.5*pHp, theta*upsilon)
       << ": rhs_part_2 = " << sigma*pi*std::max(dual_norm0, dual_norm - dual_norm0)
       << endl;
  if (model_reduction >= rhs)
    return true;
  else
    return false;
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
double FFOMWithSMART<Vec,PrimVec,DualVec>::UpdatePenaltyParameter(
    const double& grad_dot_search, const double& pHp, const double& theta,
    const double& upsilon, const double& tau, const double& dual_norm0,
    const double& dual_norm) const {
  double pi = (grad_dot_search + std::max(0.5*pHp, theta*upsilon)) /
      ((1-tau)*(dual_norm0 - dual_norm));
  return pi + 1e-4;
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
FISQPSolver<Vec,PrimVec,DualVec>::FISQPSolver() {
  maxiter_ = -1;
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FISQPSolver<Vec,PrimVec,DualVec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "FISQPSolver::SubspaceSize(): "
         << "illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  maxiter_ = m;

  // Note: STL vectors are used for V_ and Z_ while boost::numeric::ublas vectors
  // are used for double based vectors and matrices; this choice was made
  // mostly to help emphasize that the objects in V_ and Z_ are fundamentally
  // different (they are some sort of vectors themselves).
  V_.reserve(maxiter_+1);
  Z_.reserve(maxiter_);
  g_.resize(maxiter_+1, 0.0);
  g_tang_.resize(maxiter_+1, 0.0);
  sn_.resize(maxiter_+1, 0.0);
  cs_.resize(maxiter_+1, 0.0);
  y_.resize(maxiter_, 0.0);
  y_old_.resize(maxiter_, 0.0);
  H_.resize(maxiter_+1, maxiter_, 0.0);
  VtZ_.resize(maxiter_+1, maxiter_, 0.0);
  ZtZ_prim_.resize(maxiter_, maxiter_, 0.0);
  VtZ_prim_.resize(maxiter_+1, maxiter_, 0.0);
  VtZ_dual_.resize(maxiter_+1, maxiter_, 0.0);
  VtV_dual_.resize(maxiter_+1, maxiter_+1, 0.0);
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FISQPSolver<Vec,PrimVec,DualVec>::MemoryRequired(
    ptree& num_required) const {
  if (maxiter_ < 1) {
    cerr << "FISQPSolver::MemoryRequired(): "
         << "SubspaceSize must be called first (maxiter_ is undefined)" << endl;
    throw(-1);
  }
  num_required.put("num_vec", 2*maxiter_ + 1 /* V and Z */
                   + 1 /* residual */
                   + 1 /* wrk */);  
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FISQPSolver<Vec,PrimVec,DualVec>::Solve(
    const ptree & ptin, const Vec & b, Vec & x, 
    MatrixVectorProduct<Vec,Vec> & mat_vec, Preconditioner<Vec,Vec> & precond,
    ptree & ptout, ostream & fout) {
  V_.clear();
  Z_.clear();
  Vec res(b);
  g_ = ublas::zero_vector<double>(g_.size());
  g_tang_ = ublas::zero_vector<double>(g_tang_.size());
  sn_ = ublas::zero_vector<double>(sn_.size());
  cs_ = ublas::zero_vector<double>(cs_.size());
  y_ = ublas::zero_vector<double>(y_.size());
  y_old_ = ublas::zero_vector<double>(y_old_.size());
  H_ = ublas::zero_matrix<double>(H_.size1(),H_.size2());
  ZtZ_prim_ = ublas::zero_matrix<double>(ZtZ_prim_.size1(),ZtZ_prim_.size2());
  VtV_dual_ = ublas::zero_matrix<double>(VtV_dual_.size1(),VtV_dual_.size2());
  VtZ_prim_ = ublas::zero_matrix<double>(VtZ_prim_.size1(),VtZ_prim_.size2());
  VtZ_dual_ = ublas::zero_matrix<double>(VtZ_dual_.size1(),VtZ_dual_.size2());
  ublas::vector<double> y_comp, y_tang, y_norm;
  iters_ = 0;
  
  // calculate the norm of the rhs vector
  double grad0 = b.primal().Norm2();
  double feas0 = b.dual().Norm2();
  double norm0 = sqrt(grad0*grad0 + feas0*feas0);

  // get the scalings
  double grad_scale = ptin.get<double>("grad_scale", 1.0);
  double feas_scale = ptin.get<double>("feas_scale", 1.0);

  // calculate the initial residual (actually the negative residual)
  // and compute its norm
  if (ptin.get<bool>("dynamic", false))
    mat_vec.set_product_tol(ptin.get<double>("tol")/
                            static_cast<double>(maxiter_));
  V_.push_back(b); // needed to initialize V_.[0], unfortunately
  //mat_vec(x, V_[0]);
  //V_[0] -= b;
  V_[0].primal() *= grad_scale;
  V_[0].dual() *= feas_scale;

  double beta = V_[0].Norm2();
  double gamma = V_[0].dual().Norm2();

  // normalize residual to get v_{0}
  V_[0] /= beta;
  VtV_dual_(0,0) = InnerProd(V_[0].dual(), V_[0].dual());
    
  // initialize the RHS of the reduced systems
  g_(0) = beta;
  g_tang_(0) = grad_scale*InnerProd(V_[0].primal(), b.primal());

  // output header information including initial residual
  int i = 0;
  WriteHeader(fout, ptin.get<double>("tol"), norm0, feas0);
  double gamma_comp = gamma;
  double null_qual = 1.0;
  double res_norm = norm0;
  WriteHistory(fout, i, res_norm/norm0, gamma/(feas_scale*feas0),
               gamma_comp/(feas_scale*feas0), null_qual);

  // loop over all search directions
  double radius = ptin.get<double>("radius", 1e100)/grad_scale;
  bool lin_depend = false;
  bool neg_curv = false;
  bool step_violate = false;
  double pred = 0.0;
  double pred_comp = 0.0;
  for (i = 0; i < maxiter_; i++) {
    iters_++;

    // precondition the Vec V_[i] and store result in Z_[i]
    Z_.push_back(b);
    precond(V_[i], Z_[i]);
    
    // add to Krylov subspace
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
                              (beta*static_cast<double>(maxiter_)));
    V_.push_back(Z_[i]);
    Z_[i].primal() *= grad_scale;
    Z_[i].dual() *= feas_scale;
    mat_vec(Z_[i], V_[i+1]);
    Z_[i].primal() /= grad_scale;
    Z_[i].dual() /= feas_scale;
    V_[i+1].primal() *= grad_scale;
    V_[i+1].dual() *= feas_scale;

    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H_, V_);
    } catch (string err_msg) {
      cerr << "FISQP: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

    // compute new row and column of VtZ_ matrix
    // TODO: this is not efficient
    for (int k = 0; k <= i; k++) {
      VtZ_prim_(k,i) = InnerProd(V_[k].primal(), Z_[i].primal());
      VtZ_prim_(i+1,k) = InnerProd(V_[i+1].primal(), Z_[k].primal());
      VtZ_dual_(k,i) = InnerProd(V_[k].dual(), Z_[i].dual());
      VtZ_dual_(i+1,k) = InnerProd(V_[i+1].dual(), Z_[k].dual());
      VtZ_(k,i) = VtZ_prim_(k,i) + VtZ_dual_(k,i);
      VtZ_(i+1,k) = VtZ_prim_(i+1,k) + VtZ_dual_(i+1,k);
      ZtZ_prim_(k,i) = InnerProd(Z_[k].primal(), Z_[i].primal());
      ZtZ_prim_(i,k) = ZtZ_prim_(k,i);      
      VtV_dual_(k,i+1) = InnerProd(V_[k].dual(), V_[i+1].dual());
      VtV_dual_(i+1,k) = VtV_dual_(k,i+1);
    }
    g_tang_(i+1) = grad_scale*InnerProd(V_[i+1].primal(), b.primal());
    VtV_dual_(i+1,i+1) = InnerProd(V_[i+1].dual(), V_[i+1].dual());

    // solve the reduced problems and compute the residual
    boost::tie(beta, gamma, gamma_comp, null_qual, neg_curv, step_violate,
               pred, pred_comp)
        = SolveSubspaceProblems(i+1, radius, H_, g_, g_tang_, ZtZ_prim_, VtZ_,
                                VtZ_prim_, VtZ_dual_, VtV_dual_, y_, y_comp,
                                y_tang, y_norm);
    res_norm = pow(gamma/feas_scale,2.0) +
        (beta*beta - gamma*gamma)/(grad_scale*grad_scale);
    if (res_norm < 0.0) res_norm = 0.0;
    res_norm = sqrt(res_norm);
    WriteHistory(fout, i+1, res_norm/norm0, gamma/(feas_scale*feas0),
                 gamma_comp/(feas_scale*feas0), null_qual);

    // check for convergence
    if ( (step_violate) || (neg_curv) || (pred_comp > 2.0*pred) ) {
      if ( (null_qual < ptin.get<double>("tol")) && //(gamma < 0.5*feas0*feas_scale) &&
           (gamma_comp < feas0*feas_scale) ) break;
    } else if ( (res_norm < ptin.get<double>("tol")*norm0) &&
                (gamma < feas0*feas_scale) ) break;
  }
  
  ptout.put<int>("iters", iters_);
  ptout.put<double>("res", beta); // not valid if neg_curv || step_violate
  if ( (neg_curv) || (step_violate) || (pred_comp > 2.0*pred) ) {
    // use composite step approach
    if (neg_curv) fout << "# negative curvature detected" << endl;
    if (step_violate) fout << "# trust radius exceeded" << endl;
    if (pred_comp > 2.0*pred) {
      fout << "# composite-step predicted reduction much greater" << endl;
      fout << "# pred = " << pred << ": pred_comp = " << pred_comp << endl;
    }
    // compute solution
    x = 0.0;
    for (int k = 0; k < iters_; k++) {
      // x = x + y[k]*z[k]
      x.primal().EqualsAXPlusBY(1.0, x.primal(), y_comp[k], Z_[k].primal());
      x.dual().EqualsAXPlusBY(1.0, x.dual(), y_norm[k], Z_[k].dual());
    }   
  } else {
    // use the primal-dual (FGMRES) step
    x = 0.0;
    for (int k = 0; k < iters_; k++) {
      // x = x + y[k]*z[k]
      x.EqualsAXPlusBY(1.0, x, y_[k], Z_[k]);
    }
  }

  // scale the solution
  x.primal() *= grad_scale;
  x.dual() *= feas_scale;
  
  if (ptin.get<bool>("check", false)) {
    // recalculate explicilty and check final residual
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(0.1*ptin.get<double>("tol")/
                              static_cast<double>(maxiter_));
    mat_vec(x, res);
    // V_[0] = b - V_[0] = b - A*x
    res.EqualsAXPlusBY(1.0, b, -1.0, res);
    double true_res = res.Norm2();
    double true_feas = res.dual().Norm2();
    fout << "# FISQP final (true) residual : |res|/|res0| = "
         << true_res/norm0 << endl;
    fout << "# FISQP final (true) feas.  : |feas|/|feas0| = "
         << true_feas/feas0 << endl;
    ptout.put<double>("res", true_res);
    double computed_feas = gamma/feas_scale;
    if ( (neg_curv) || (step_violate) || (pred_comp > 2.0*pred) )
      computed_feas = gamma_comp/feas_scale;
    if (fabs(true_feas - computed_feas) >  0.01*ptin.get<double>("tol")*feas0) {
      fout << "# WARNING in FISQP: true constraint norm and calculated "
           << "constraint norm do not agree." << endl;
      fout << "# (feas_true - feas_comp)/feas0 = "
           << (true_feas - computed_feas)/feas0 << endl;
    }
    if ( (!neg_curv) && (!step_violate) && (pred_comp <= 2.0*pred) ) {
      if (fabs(true_res - res_norm) > 0.01*ptin.get<double>("tol")*norm0) {
        fout << "# WARNING in FISQP: true residual norm and calculated "
             << "residual norm do not agree." << endl;
        fout << "# (res - beta)/res0 = "
             << (true_res - res_norm)/norm0 << endl;
      }
    }
  }
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FISQPSolver<Vec,PrimVec,DualVec>::ReSolve(
    const ptree & ptin, const Vec & b, Vec & x, ptree & ptout, ostream & fout) {
  // calculate the norm of the rhs vector
  double grad0 = b.primal().Norm2();
  double feas0 = b.dual().Norm2();
  double norm0 = sqrt(grad0*grad0 + feas0*feas0);

  // get the scalings
  double grad_scale = ptin.get<double>("grad_scale", 1.0);
  double feas_scale = ptin.get<double>("feas_scale", 1.0);
  
  // solve the reduced problems and compute the residual
  ublas::vector<double> y_comp, y_tang, y_norm;
  double radius = ptin.get<double>("radius", 1e100)/grad_scale;
  double beta, gamma, gamma_comp, null_qual;
  bool neg_curv, step_violate;
  double pred, pred_comp;
  boost::tie(beta, gamma, gamma_comp, null_qual, neg_curv, step_violate,
             pred, pred_comp)
      = SolveSubspaceProblems(iters_, radius, H_, g_, g_tang_, ZtZ_prim_, VtZ_,
                              VtZ_prim_, VtZ_dual_, VtV_dual_, y_, y_comp,
                              y_tang, y_norm);
  double res_norm = pow(gamma/feas_scale,2.0) +
      (beta*beta - gamma*gamma)/(grad_scale*grad_scale);
  if (res_norm < 0.0) res_norm = 0.0;
  res_norm = sqrt(res_norm);
  fout << "# FISQP: resolving at new radius" << endl;
  WriteHistory(fout, iters_+1, res_norm/norm0, gamma/(feas_scale*feas0),
               gamma_comp/(feas_scale*feas0), null_qual);
  if (neg_curv) fout << "# negative curvature detected" << endl;
  if (step_violate) fout << "# trust radius exceeded" << endl;
  if (pred_comp < 2.0*pred) fout << "# composite-step predicted reduction greater"
                                 << endl;
  // always use composite step approach during resolve
  x = 0.0;
  for (int k = 0; k < iters_; k++) {
    // x = x + y[k]*z[k]
    x.primal().EqualsAXPlusBY(1.0, x.primal(), y_comp[k], Z_[k].primal());
    x.dual().EqualsAXPlusBY(1.0, x.dual(), y_norm[k], Z_[k].dual());
  }
  // scale the solution
  x.primal() *= grad_scale;
  x.dual() *= feas_scale; 
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
boost::tuple<double, double, double, double, bool, bool, double, double> 
FISQPSolver<Vec,PrimVec,DualVec>::SolveSubspaceProblems(
    const int& iter, const double& radius, const ublas::matrix<double>& H,
    const ublas::vector<double>& g, const ublas::vector<double>& g_tang,
    const ublas::matrix<double>& ZtZ_prim, const ublas::matrix<double>& VtZ,
    const ublas::matrix<double>& VtZ_prim, const ublas::matrix<double>& VtZ_dual,
    const ublas::matrix<double>& VtV_dual, ublas::vector<double>& y,
    ublas::vector<double>& y_comp, ublas::vector<double>& y_tang,
    ublas::vector<double>& y_norm) {
  
  // these ranges help improve readability below
  ublas::vector_range<ublas::vector<double> >
      y_r(y_, ublas::range(0, iter)),
      g_r(g_, ublas::range(0, iter+1));
  ublas::matrix_range<const ublas::matrix<double> >
      ZtZ_prim_r(ZtZ_prim, ublas::range(0,iter), ublas::range(0,iter)),
      VtZ_r(VtZ_, ublas::range(0,iter+1), ublas::range(0,iter)),
      VtZ_prim_r(VtZ_prim, ublas::range(0,iter+1), ublas::range(0,iter)),
      VtZ_dual_r(VtZ_dual, ublas::range(0,iter+1), ublas::range(0,iter)),
      VtV_dual_r(VtV_dual, ublas::range(0,iter+1), ublas::range(0,iter+1)),
      H_r(H_, ublas::range(0,iter+1), ublas::range(0,iter));
  
  // solve the reduced primal-dual problem and compute residual
  solveLeastSquares(iter+1, iter, H, g, y); 
  ublas::vector<double> res_red = ublas::prod(H_r, y_r) - g_r;
  double beta = norm_2(res_red);
  double gamma = ublas::inner_prod(res_red, ublas::prod(VtV_dual_r, res_red));
  if (gamma < 0.0)
    gamma = 0.0;
  else
    gamma = sqrt(gamma);

  // solve the reduced tangential-step problem
  y_tang.resize(iter);  
  //solveReduced(iter, H, g_tang, y_tang); // project onto V
  solveLeastSquares(iter+1, iter, H, g_tang, y_tang);

  // find the Hessian in the reduced space and some inner products
  ublas::matrix<double> Hred = ublas::prod(ublas::trans(VtZ_r), H_r)
      - ublas::prod(ublas::trans(VtZ_dual_r), H_r)
      - ublas::prod(ublas::trans(H_r), VtZ_dual_r);
  double ytHy_tang = ublas::inner_prod(y_tang, ublas::prod(Hred, y_tang));
  double ytZtZy_tang = ublas::inner_prod(y_tang, ublas::prod(ZtZ_prim_r,y_tang));
  double ytZtZy = ublas::inner_prod(y_r, ublas::prod(ZtZ_prim_r, y_r));

  // compute the normal step
  double alpha = ublas::inner_prod(y_r, ublas::prod(ZtZ_prim_r, y_tang))
      / ytZtZy_tang;
  y_norm.resize(iter);
  y_norm = y_r - alpha*y_tang;

  // scale the normal step if necessary
  double ytZtZy_norm =
      ublas::inner_prod(y_norm, ublas::prod(ZtZ_prim_r, y_norm));
  if (sqrt(ytZtZy_norm) > 0.8*radius) {
    y_norm *= 0.8*radius/sqrt(ytZtZy_norm);
    ytZtZy_norm = 0.8*0.8*radius*radius;
  }

  // solve the tangential step problem
  double t = 0.0;
  if (ytHy_tang >= kEpsilon*ytZtZy_tang) { // rhs was sqrt(kEpsilon)*ytZtZ_tang
    for (int k = 0; k < iter; k++)
      t += g(0)*VtZ_prim_r(0,k)*y_tang(k); // -(nabla L)^T p_tang
    t += -ublas::inner_prod(y_norm, ublas::prod(Hred, y_tang));
    t /= ytHy_tang; 
  }

  // if trust radius is violated, or negative/zero curvature...
  if ( (ytHy_tang < kEpsilon*ytZtZy_tang) ||
       (sqrt(ytZtZy_norm) + fabs(t)*sqrt(ytZtZy_tang) > radius) ) {
    // ...minimum must be at edge of trust radius
    t = (radius - sqrt(ytZtZy_norm))/sqrt(ytZtZy_tang);
    // if necessary, make step a descent direction for Lagrangian
    double grad_dot_tang = 0.0;
    for (int k = 0; k < iter; k++) grad_dot_tang += VtZ_prim(0,k)*y_tang(k);
    if (grad_dot_tang < 0.0)
      t *= -1.0; 
  }

  // compute the composite step and the constraint residual
  y_comp.resize(iter);
  y_tang *= t;
  y_comp = y_tang + y_norm;
  res_red = ublas::prod(H_r, y_comp) - g_r;
  double gamma_comp = ublas::inner_prod(
      res_red, ublas::prod(VtV_dual_r, res_red));
  if (gamma_comp < 0.0)
    gamma_comp = 0.0;
  else
    gamma_comp = sqrt(gamma_comp);

  // compute the null-vector quality
  ublas::vector<double> VtVHy_tang = ublas::prod(ublas::prod(VtV_dual_r, H_r),
                                                 y_tang);
  double yHtVtVHy_tang = ublas::inner_prod(y_tang, ublas::prod(
      ublas::trans(H_r), VtVHy_tang));
  double null_qual = yHtVtVHy_tang/(t*t*ytZtZy_tang);
  if (null_qual < 0.0)
    null_qual = 0.0;
  else
    null_qual = sqrt(null_qual);

  // compute the predicted reductions in the objective
  double pred = 0.0;
  for (int k = 0; k < iter; k++) pred += g(0)*VtZ_prim_r(0,k)*y_r(k);
  pred += -0.5*ublas::inner_prod(y_r, ublas::prod(Hred, y_r));
  double pred_comp = 0.0;
  for (int k = 0; k < iter; k++) pred_comp += g(0)*VtZ_prim_r(0,k)*y_comp(k);
  pred_comp += -0.5*ublas::inner_prod(y_comp, ublas::prod(Hred, y_comp));
  
  // return residuals and booleans
  bool neg_curv = false;
  if (ytHy_tang < kEpsilon*ytZtZy_tang) neg_curv = true;
  bool step_violate = false;
  if (sqrt(ytZtZy) > radius) step_violate = true;
  return boost::make_tuple(beta, gamma,  gamma_comp, null_qual, neg_curv,
                           step_violate, pred, pred_comp);
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FISQPSolver<Vec,PrimVec,DualVec>::WriteHeader(
    ostream& os, const double& tol, const double& res0, const double& feas0) {
#if 0
  if (!(os.good())) {
    cerr << "FISQPSolver::WriteHeader(): "
	 << "ostream is not good for i/o operations." << endl;
    throw(-1);
  }
#endif
  os << "# FISQP convergence history" << endl;
  os << boost::format(
      "# residual tolerance target %|30t| = %|-10.6|\n") % tol;
  os << boost::format(
      "# initial residual norm %|30t| = %|-10.6|\n") % res0;
  os << boost::format(
      "# initial constraint norm %|30t| = %|-10.6|\n") % feas0;
  os << boost::format(string("%|6| %|8t| %|-12| %|20t| %|-12|")+
                      string(" %|32t| %|-12| %|44t| %|-12|\n"))
      % "# iter" % "rel. res." % "rel. feas." % "comp. feas" % "null qual.";
  os.flush();
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FISQPSolver<Vec,PrimVec,DualVec>::WriteHistory(
    ostream& os, const int& iter, const double& res, const double& feas,
    const double& feas_comp, const double& null_qual) {
#if 0
  if (!(os.good())) {
    cerr << "FISQPSolver::WriteHistory(): "
	 << "ostream is not good for i/o operations." << endl;
    throw(-1);
  }
#endif
  os << boost::format(string("%|6| %|8t| %|-12.6| %|20t| %|-12.6|")+
                      string(" %|32t| %|-12.6| %|44t| %|-12.6|\n"))
      % iter % res % feas % feas_comp % null_qual;
  os.flush();
}
// ==============================================================================
#if 0
template <class Vec, class PrimVec, class DualVec>
void FISQPSolver<Vec,PrimVec,DualVec>::Correct2ndOrder(
    const ptree & ptin, const DualVec & ceq, const Vec & b, Vec & x,
    ptree & ptout, ostream & fout) {

  // calculate the norm of the rhs vector
  // TODO: save this from before
  double norm0 = b.Norm2();
  double dual_norm0 = b.dual().Norm2();
  
  // update the reduced-system RHS
  res_->dual() -= ceq; // VERY IMPORTANT; ceq is acctually x.dual()
  cout << "updating reduced-system RHS..." << endl;
  cout << "g_.size() = " << g_.size() << endl;
  cout << "iters_ = " << iters_ << endl;
  for (int k = 0; k < iters_+1; k++)
    g_(k) += InnerProd(res_->dual(), V_[k].dual());
  
  // compute the 2nd-order step
  cout <<"before solveLeastSquares..." << endl;
  solveLeastSquares(iters_+1, iters_, H_, g_, y_);
  res_->primal() = x.primal(); // save the objective gradient
  x = 0.0;
  cout <<"before computing x..." << endl;
  for (int k = 0; k < iters_; k++) {
    // x = x + y[k]*z[k]
    x.EqualsAXPlusBY(1.0, x, y_[k], Z_[k]);
  }
  double grad_dot_step = InnerProd(res_->primal(), x.primal());

  // compute residual
  ublas::vector_range<ublas::vector<double> >
      y_r(y_, ublas::range(0, iters_)),
      g_r(g_, ublas::range(0, iters_+1));
  ublas::matrix_range<ublas::matrix<double> >
      H_r(H_, ublas::range(0,iters_+1), ublas::range(0,iters_));
  cout <<"before res_red..." << endl;
  ublas::vector<double> res_red = g_r - ublas::prod(H_r, y_r);
  res_->primal() = 0.0;
  cout <<"before computing residual..." << endl;
  for (int k = 0; k < iters_+1; k++)
    res_->EqualsAXPlusBY(1.0, *res_, res_red(k), V_[k]);
  double beta = res_->Norm2();
  double gamma = res_->dual().Norm2();
  fout << "# second-order corrected step" << endl;
  writeKrylovHistory(fout, iters_+1, beta, norm0);  

  // update penalty parameter, if necessary  
  double pi = ptin.get<double>("pi");
  double ytHy = 0.0;
  UpdatePenaltyParameter(grad_dot_step, ytHy, ptin.get<double>("tau", 0.2),
                         dual_norm0, gamma, pi);  
  ptout.put<double>("pi", pi);
  ptout.put<double>("model_reduction", -grad_dot_step + pi*(dual_norm0-gamma));

  cout << "pi = " << pi << endl;
  cout << "grad_dot_step =" << grad_dot_step << endl;
  cout << "(dual_norm-gamma) = " << dual_norm0 - gamma << endl;
  cout << "norm2(y) = " << norm_2(y_r) << endl;
  //cout << "norm2(g) = " << res_->primal().Norm2() << endl;  
}
#endif
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
FLECSSolver<Vec,PrimVec,DualVec>::FLECSSolver() {
  maxiter_ = -1;
  mu_ = 0.0;
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FLECSSolver<Vec,PrimVec,DualVec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "FLECSSolver::SubspaceSize(): "
         << "illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  maxiter_ = m;

  // Note: STL vectors are used for V_ and Z_ while boost::numeric::ublas vectors
  // are used for double based vectors and matrices; this choice was made
  // mostly to help emphasize that the objects in V_ and Z_ are fundamentally
  // different (they are some sort of vectors themselves).
  V_.reserve(maxiter_+1);
  Z_.reserve(maxiter_);
  g_.resize(maxiter_+1, 0.0);
  y_.resize(maxiter_, 0.0);
  H_.resize(maxiter_+1, maxiter_, 0.0);
  VtZ_.resize(maxiter_+1, maxiter_, 0.0);
  ZtZ_prim_.resize(maxiter_, maxiter_, 0.0);
  VtZ_prim_.resize(maxiter_+1, maxiter_, 0.0);
  VtZ_dual_.resize(maxiter_+1, maxiter_, 0.0);
  VtV_dual_.resize(maxiter_+1, maxiter_+1, 0.0);
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FLECSSolver<Vec,PrimVec,DualVec>::MemoryRequired(
    ptree& num_required) const {
  if (maxiter_ < 1) {
    cerr << "FLECSSolver::MemoryRequired(): "
         << "SubspaceSize must be called first (maxiter_ is undefined)" << endl;
    throw(-1);
  }
  num_required.put("num_vec", 2*maxiter_ + 1 /* V and Z */ + 1 /* res */);  
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FLECSSolver<Vec,PrimVec,DualVec>::Solve(
    const ptree & ptin, const Vec & b, Vec & x, 
    MatrixVectorProduct<Vec,Vec> & mat_vec, Preconditioner<Vec,Vec> & precond,
    ptree & ptout, ostream & fout) {
  V_.clear();
  Z_.clear();
  Vec res(b);
  g_ = ublas::zero_vector<double>(g_.size());
  y_ = ublas::zero_vector<double>(y_.size());
  H_ = ublas::zero_matrix<double>(H_.size1(),H_.size2());
  ZtZ_prim_ = ublas::zero_matrix<double>(ZtZ_prim_.size1(),ZtZ_prim_.size2());
  VtV_dual_ = ublas::zero_matrix<double>(VtV_dual_.size1(),VtV_dual_.size2());
  VtZ_prim_ = ublas::zero_matrix<double>(VtZ_prim_.size1(),VtZ_prim_.size2());
  VtZ_dual_ = ublas::zero_matrix<double>(VtZ_dual_.size1(),VtZ_dual_.size2());
  ublas::vector<double> y_aug, y_mult;
  iters_ = 0;
  
  // calculate the norm of the rhs vector
  double grad0 = b.primal().Norm2();
  double feas0 = b.dual().Norm2();
  double norm0 = sqrt(grad0*grad0 + feas0*feas0);

  // get the scalings
  double grad_scale = ptin.get<double>("grad_scale", 1.0);
  double feas_scale = ptin.get<double>("feas_scale", 1.0);

  // calculate the initial residual (actually the negative residual)
  // and compute its norm
  if (ptin.get<bool>("dynamic", false))
    mat_vec.set_product_tol(ptin.get<double>("tol")/
                            static_cast<double>(maxiter_));
  V_.push_back(b); // needed to initialize V_.[0], unfortunately
  //mat_vec(x, V_[0]);
  //V_[0] -= b;
  V_[0].primal() *= grad_scale;
  V_[0].dual() *= feas_scale;

  // normalize residual to get v_{0}
  double beta = V_[0].Norm2();
  V_[0] /= beta;
  VtV_dual_(0,0) = InnerProd(V_[0].dual(), V_[0].dual());
  double gamma = beta*sqrt(std::max(VtV_dual_(0,0),0.0));
  double omega = sqrt(std::max(beta*beta - gamma*gamma, 0.0));
    
  // initialize the RHS of the reduced system
  g_(0) = beta;

  // output header information including initial residual
  int i = 0;
  WriteHeader(fout, ptin.get<double>("tol"), norm0, grad0, feas0);
  double beta_aug = beta;
  double gamma_aug = gamma;
  double res_norm = norm0;
  double pred = 0.0;
  double pred_aug = 0.0;
  mu_ = ptin.get<double>("mu", 1.0);
  WriteHistory(fout, i, res_norm/norm0, omega/(grad_scale*grad0),
               gamma/(feas_scale*feas0), gamma_aug/(feas_scale*feas0),
               pred, pred_aug);

  // loop over all search directions
  double radius = ptin.get<double>("radius", 1e100)/grad_scale;
  bool lin_depend = false;
  bool neg_curv = false;
  bool trust_active = false;
  for (i = 0; i < maxiter_; i++) {
    iters_++;

    // precondition the Vec V_[i] and store result in Z_[i]
    Z_.push_back(b);
    precond(V_[i], Z_[i]);
    
    // add to Krylov subspace
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
                              (beta*static_cast<double>(maxiter_)));
    V_.push_back(Z_[i]);
    Z_[i].primal() *= grad_scale;
    Z_[i].dual() *= feas_scale;
    mat_vec(Z_[i], V_[i+1]);
    Z_[i].primal() /= grad_scale;
    Z_[i].dual() /= feas_scale;
    V_[i+1].primal() *= grad_scale;
    V_[i+1].dual() *= feas_scale;

    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H_, V_);
    } catch (string err_msg) {
      cerr << "FLECS: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

    // compute new row and column of VtZ_ matrix
    // TODO: this is not efficient
    for (int k = 0; k <= i; k++) {
      VtZ_prim_(k,i) = InnerProd(V_[k].primal(), Z_[i].primal());
      VtZ_prim_(i+1,k) = InnerProd(V_[i+1].primal(), Z_[k].primal());
      VtZ_dual_(k,i) = InnerProd(V_[k].dual(), Z_[i].dual());
      VtZ_dual_(i+1,k) = InnerProd(V_[i+1].dual(), Z_[k].dual());
      VtZ_(k,i) = VtZ_prim_(k,i) + VtZ_dual_(k,i);
      VtZ_(i+1,k) = VtZ_prim_(i+1,k) + VtZ_dual_(i+1,k);
      ZtZ_prim_(k,i) = InnerProd(Z_[k].primal(), Z_[i].primal());
      ZtZ_prim_(i,k) = ZtZ_prim_(k,i);      
      VtV_dual_(k,i+1) = InnerProd(V_[k].dual(), V_[i+1].dual());
      VtV_dual_(i+1,k) = VtV_dual_(k,i+1);
    }
    VtV_dual_(i+1,i+1) = InnerProd(V_[i+1].dual(), V_[i+1].dual());

    // solve the reduced problems and compute the residual
#if 0
    mu_ = ptin.get<double>("mu", 1.0)/10.0;
    do {
      mu_ *= 10.0;
      boost::tie(beta, gamma, omega, beta_aug, gamma_aug, neg_curv, trust_active,
                 pred, pred_aug)
          = SolveSubspaceProblems(i+1, radius, H_, g_, mu_, ZtZ_prim_, VtZ_,
                                  VtZ_prim_, VtZ_dual_, VtV_dual_, y_, y_aug,
                                  y_mult);
    } while (gamma_aug > feas0*feas_scale);
#endif
    boost::tie(beta, gamma, omega, beta_aug, gamma_aug, neg_curv, trust_active,
                 pred, pred_aug)
          = SolveSubspaceProblems(i+1, radius, H_, g_, mu_, ZtZ_prim_, VtZ_,
                                  VtZ_prim_, VtZ_dual_, VtV_dual_, y_, y_aug,
                                  y_mult);

    res_norm = pow(gamma/feas_scale,2.0) +
        (beta*beta - gamma*gamma)/(grad_scale*grad_scale);
    res_norm = sqrt(std::max(res_norm, 0.0));
    WriteHistory(fout, i+1, res_norm/norm0, omega/(grad_scale*grad0),
                 gamma/(feas_scale*feas0), gamma_aug/(feas_scale*feas0), pred,
                 pred_aug);

    // check for convergence
    if ( (gamma < ptin.get<double>("tol")*feas0*feas_scale) &&
         (omega < ptin.get<double>("tol")*grad0*grad_scale) ) break;
    //if (res_norm < ptin.get<double>("tol")*norm0) break;
         //(gamma < ptin.get<double>("tol")*feas0*feas_scale) &&
         //(gamma_aug < feas0*feas_scale) ) break;
  }
  
  ptout.put<int>("iters", iters_);
  ptout.put<double>("res", beta); // not valid if neg_curv || step_violate
  ptout.put<double>("active", trust_active);

  if (neg_curv) fout << "# negative curvature suspected" << endl;
  if (trust_active) fout << "# trust-radius constraint active" << endl;

  // compute solution: use augmented-Lagrangian step for primal, FGMRES for dual
  x = 0.0;
  for (int k = 0; k < iters_; k++) {
    // x = x + y[k]*z[k]
    x.primal().EqualsAXPlusBY(1.0, x.primal(), y_aug[k], Z_[k].primal());
    x.dual().EqualsAXPlusBY(1.0, x.dual(), y_mult[k], Z_[k].dual());
  }

  // scale the solution
  x.primal() *= grad_scale;
  x.dual() *= feas_scale;
  
  if (ptin.get<bool>("check", false)) {
    // recalculate explicilty and check final residual
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(0.1*ptin.get<double>("tol")/
                              static_cast<double>(maxiter_));
    V_[0] = 0;
    for (int k = 0; k < iters_; k++)
      V_[0].EqualsAXPlusBY(1.0, V_[0], y_mult[k], Z_[k]);
    //mat_vec(x, res);
    mat_vec(V_[0], res);
    // V_[0] = b - V_[0] = b - A*x
    res.EqualsAXPlusBY(1.0, b, -1.0, res);
    double true_res = res.Norm2();
    double true_feas = res.dual().Norm2();
    fout << "# FLECS final (true) rel. res.  : |res|/|res0| = "
         << true_res/norm0 << endl;
    fout << "# FLECS final (true) rel. feas. : |feas|/|feas0| = "
         << true_feas/feas0 << endl;
    ptout.put<double>("res", true_res);
    double computed_feas = gamma/feas_scale;
    //if ( (neg_curv) || (step_violate) )
    //  computed_feas = gamma_aug/feas_scale;
    if (fabs(true_feas - computed_feas) >  0.01*ptin.get<double>("tol")*feas0) {
      fout << "# WARNING in FLECS: true constraint norm and calculated "
           << "constraint norm do not agree." << endl;
      fout << "# (feas_true - feas_comp)/feas0 = "
           << (true_feas - computed_feas)/feas0 << endl;
    }
    //if ( (!neg_curv) && (!step_violate) ) {
    if (fabs(true_res - res_norm) > 0.01*ptin.get<double>("tol")*norm0) {
      fout << "# WARNING in FLECS: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = "
             << (true_res - res_norm)/norm0 << endl;
    }
    //}
  }
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FLECSSolver<Vec,PrimVec,DualVec>::ReSolve(
    const ptree & ptin, const Vec & b, Vec & x, ptree & ptout, ostream & fout) {
  // calculate the norm of the rhs vector
  double grad0 = b.primal().Norm2();
  double feas0 = b.dual().Norm2();
  double norm0 = sqrt(grad0*grad0 + feas0*feas0);

  // get the scalings
  double grad_scale = ptin.get<double>("grad_scale", 1.0);
  double feas_scale = ptin.get<double>("feas_scale", 1.0);
  
  // solve the reduced problems and compute the residual
  ublas::vector<double> y_aug, y_mult;
  double radius = ptin.get<double>("radius", 1e100)/grad_scale;
  //double mu = ptin.get<double>("mu", 10.0);  
  double beta, gamma, omega, beta_aug, gamma_aug;
  bool neg_curv, trust_active;
  double pred, pred_aug;
  boost::tie(beta, gamma, omega, beta_aug, gamma_aug, neg_curv, trust_active,
             pred, pred_aug)
      = SolveSubspaceProblems(iters_, radius, H_, g_, mu_, ZtZ_prim_, VtZ_,
                              VtZ_prim_, VtZ_dual_, VtV_dual_, y_, y_aug,
                              y_mult);
  double res_norm = pow(gamma/feas_scale,2.0) +
      (beta*beta - gamma*gamma)/(grad_scale*grad_scale);
  res_norm = sqrt(std::max(res_norm, 0.0));
  fout << "# FLECS: resolving at new radius" << endl;
  WriteHistory(fout, iters_+1, res_norm/norm0, omega/(grad_scale*grad0),
               gamma/(feas_scale*feas0), gamma_aug/(feas_scale*feas0), pred,
               pred_aug);
  if (neg_curv) fout << "# negative curvature suspected" << endl;
  if (trust_active) fout << "# trust-radius constraint active" << endl;
  ptout.put<double>("active", trust_active);
  // always use composite step approach during resolve
  x = 0.0;
  for (int k = 0; k < iters_; k++) {
    // x = x + y[k]*z[k]
    x.primal().EqualsAXPlusBY(1.0, x.primal(), y_aug[k], Z_[k].primal());
    x.dual().EqualsAXPlusBY(1.0, x.dual(), y_mult[k], Z_[k].dual());
  }
  // scale the solution
  x.primal() *= grad_scale;
  x.dual() *= feas_scale;
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
boost::tuple<double, double, double, double, double, bool, bool, double, double>
FLECSSolver<Vec,PrimVec,DualVec>::SolveSubspaceProblems(
    const int& iter, const double& radius, const ublas::matrix<double>& H,
    const ublas::vector<double>& g, const double& mu, 
    const ublas::matrix<double>& ZtZ_prim, const ublas::matrix<double>& VtZ,
    const ublas::matrix<double>& VtZ_prim, const ublas::matrix<double>& VtZ_dual,
    const ublas::matrix<double>& VtV_dual, ublas::vector<double>& y,
    ublas::vector<double>& y_aug, ublas::vector<double>& y_mult) {
  
  // these ranges help improve readability below
  ublas::vector_range<ublas::vector<double> >
      y_r(y_, ublas::range(0, iter)),
      g_r(g_, ublas::range(0, iter+1));
  ublas::matrix_range<const ublas::matrix<double> >
      ZtZ_prim_r(ZtZ_prim, ublas::range(0,iter), ublas::range(0,iter)),
      VtZ_r(VtZ_, ublas::range(0,iter+1), ublas::range(0,iter)),
      VtZ_prim_r(VtZ_prim, ublas::range(0,iter+1), ublas::range(0,iter)),
      VtZ_dual_r(VtZ_dual, ublas::range(0,iter+1), ublas::range(0,iter)),
      VtV_dual_r(VtV_dual, ublas::range(0,iter+1), ublas::range(0,iter+1)),
      H_r(H_, ublas::range(0,iter+1), ublas::range(0,iter));
  
  // solve the reduced (primal-dual) problem and compute residuals
  // i.e. the FGMRES solution
  solveLeastSquares(iter+1, iter, H, g, y); 
  ublas::vector<double> res_red = ublas::prod(H_r, y_r) - g_r;
  double beta = norm_2(res_red);
  double gamma = ublas::inner_prod(res_red, ublas::prod(VtV_dual_r, res_red));
  double omega = -gamma;
  gamma = sqrt(std::max(gamma, 0.0));
  omega = sqrt(std::max(ublas::inner_prod(res_red, res_red) + omega, 0.0));

#if 0
  // check the length of the FGMRES step
  double ytZtZy = ublas::inner_prod(y_r, ublas::prod(ZtZ_prim_r, y_r));
  double fgmres_step_len = sqrt(std::max(ytZtZy, 0.0));
  bool step_violate = false;
  if (fgmres_step_len > radius) step_violate = true;
#endif

  // find the Hessian of the objective and the Hessian of the augmented
  // Lagrangian in the reduced space
  ublas::matrix<double> Hess_red = ublas::prod(ublas::trans(VtZ_r), H_r)
      - ublas::prod(ublas::trans(VtZ_dual_r), H_r)
      - ublas::prod(ublas::trans(H_r), VtZ_dual_r);
  ublas::matrix<double> VtVH = ublas::prod(VtV_dual_r, H_r);
  ublas::matrix<double> Hess_aug = mu*ublas::prod(ublas::trans(H_r), VtVH);
  Hess_aug += Hess_red;

  // compute the rhs for the augmented-Lagrangian problem
  ublas::vector<double> rhs_aug(iter, 0.0);
  for (int k = 0; k < iter; k++)
    rhs_aug(k) = -g(0)*(VtZ_prim(0,k) + mu*VtVH(0,k));
  
  double lambda = 0.0, tmp = 0.0;
  double radius_aug = radius;
  y_aug.resize(iter);
  try {
    // compute the transformation to apply trust-radius directly
    ublas::vector<double> UTU;
    factorCholesky(iter, ZtZ_prim, UTU);
    ublas::vector<double> rhs_tmp(rhs_aug), vec_tmp(iter, 0.0);
    solveU(iter, UTU, rhs_tmp, rhs_aug, true);
    for (int j = 0; j < iter; j++) {
      for (int i = 0; i < iter; i++) rhs_tmp(i) = Hess_aug(i,j);
      solveU(iter, UTU, rhs_tmp, vec_tmp, true);
      for (int i = 0; i < iter; i++) Hess_aug(i,j) = vec_tmp(i);
    }
    for (int j = 0; j < iter; j++) {
      for (int i = 0; i < iter; i++) rhs_tmp(i) = Hess_aug(j,i);
      solveU(iter, UTU, rhs_tmp, vec_tmp, true);
      for (int i = 0; i < iter; i++) Hess_aug(j,i) = vec_tmp(i);
    }
    solveTrustReduced(iter, Hess_aug, radius_aug, rhs_aug, vec_tmp, lambda, tmp);
    solveU(iter, UTU, vec_tmp, y_aug, false);    
  } catch (bool fail) {
    // factorCholesky failed; compute a conservative radius for the
    // reduced-space trust-region problem
    ublas::vector<double> eig(iter, 0.0);
    eigenvalues(iter, ZtZ_prim, eig);
    radius_aug = radius/sqrt(eig(iter-1)); // should check for zero eig
    solveTrustReduced(iter, Hess_aug, radius_aug, rhs_aug, y_aug, lambda, tmp);
  }

  // check if the trust-radius constraint is active
  bool trust_active = false;
  if (lambda > 0.0) trust_active = true;
  
  // compute the residual norms for the augmented-Lagrangian solution
  res_red = ublas::prod(H_r, y_aug) - g_r;
  double beta_aug = norm_2(res_red);
  double gamma_aug = ublas::inner_prod(res_red, ublas::prod(VtV_dual_r,
                                                            res_red));
  gamma_aug = sqrt(std::max(gamma_aug, 0.0));

  // set the dual reduced-space solution
  y_mult.resize(iter);
  y_mult = y_r;
  
  // compute the predicted reductions in the objective (not the penalty func.)
  double pred = -0.5*ublas::inner_prod(y_r, ublas::prod(Hess_red, y_r));
  for (int k = 0; k < iter; k++) pred += g(0)*VtZ_prim_r(0,k)*y_r(k);
  double pred_aug = -0.5*ublas::inner_prod(y_aug, ublas::prod(Hess_red, y_aug));
  for (int k = 0; k < iter; k++) pred_aug += g(0)*VtZ_prim_r(0,k)*y_aug(k);

  // determine if negative curvature may be present
  bool neg_curv = false;
  if (pred_aug - pred > 0.05*fabs(pred)) neg_curv = true;
  
  // return residuals and booleans
  return boost::make_tuple(beta, gamma, omega, beta_aug, gamma_aug, neg_curv,
                           trust_active, pred, pred_aug);
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FLECSSolver<Vec,PrimVec,DualVec>::Correct2ndOrder(
    const ptree & ptin, const DualVec & ceq, const Vec & b, Vec & x,
    ptree & ptout, ostream & fout) {

  // these ranges help improve readability below
  ublas::matrix_range<const ublas::matrix<double> >
      ZtZ_prim_r(ZtZ_prim_, ublas::range(0,iters_), ublas::range(0,iters_)),
      VtV_dual_r(VtV_dual_, ublas::range(0,iters_), ublas::range(0,iters_+1)),
      H_r(H_, ublas::range(0,iters_+1), ublas::range(0,iters_));
  
  // construct subspace problem
  ublas::matrix<double> VtVH = ublas::prod(VtV_dual_r, H_r);
  ublas::matrix<double> A = ZtZ_prim_r + VtVH + ublas::trans(VtVH);
  ublas::vector<double> rhs(iters_, 0.0);
  for (int k = 0; k < iters_; k++)
    rhs(k) = InnerProd(V_[k].dual(), ceq);
  solveReduced(iters_, A, rhs, y_);

  // construct the primal solution (leave dual solution unchanged)
  x.primal() = 0.0;
  for (int k = 0; k < iters_; k++)
    x.primal().EqualsAXPlusBY(1.0, x.primal(), y_(k), Z_[k].primal());
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FLECSSolver<Vec,PrimVec,DualVec>::WriteHeader(
    ostream& os, const double& tol, const double& res0, const double& grad0,
    const double& feas0) {
#if 0
  if (!(os.good())) {
    cerr << "FLECSSolver::WriteHeader(): "
	 << "ostream is not good for i/o operations." << endl;
    throw(-1);
  }
#endif
  os << "# FLECS convergence history" << endl;
  os << boost::format(
      "# residual tolerance target %|30t| = %|-10.6|\n") % tol;
  os << boost::format(
      "# initial residual norm %|30t| = %|-10.6|\n") % res0;
  os << boost::format(
      "# initial gradient norm %|30t| = %|-10.6|\n") % grad0;
  os << boost::format(
      "# initial constraint norm %|30t| = %|-10.6|\n") % feas0;
  os << boost::format(string("%|6| %|8t| %|-12| %|20t| %|-12|")+
                      string(" %|32t| %|-12| %|44t| %|-12|")+
                      string(" %|56t| %|-12| %|68t| %|-12|")+
                      string(" %|80t| %|-12|\n"))
      % "# iter" % "rel. res." % "rel. grad." % "rel. feas."
      % "aug. feas" % "pred." % "pred. aug." % "mu";
  os.flush();
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void FLECSSolver<Vec,PrimVec,DualVec>::WriteHistory(
    ostream& os, const int& iter, const double& res, const double& grad,
    const double& feas, const double& feas_aug, const double& pred,
    const double& pred_aug) {
#if 0
  if (!(os.good())) {
    cerr << "FLECSSolver::WriteHistory(): "
	 << "ostream is not good for i/o operations." << endl;
    throw(-1);
  }
#endif
  os << boost::format(string("%|6| %|8t| %|-12.6| %|20t| %|-12.6|")+
                      string(" %|32t| %|-12.6| %|44t| %|-12.6|")+
                      string(" %|56t| %|-12.6| %|68t| %|-12.6|")+
                      string(" %|80t| %|-12.6|\n"))
      % iter % res % grad % feas % feas_aug % pred % pred_aug % mu_;
  os.flush();
}
// ==============================================================================
#if 0
template <class Vec, class PrimVec, class DualVec>
void FLECSSolver<Vec,PrimVec,DualVec>::Correct2ndOrder(
    const ptree & ptin, const DualVec & ceq, const Vec & b, Vec & x,
    ptree & ptout, ostream & fout) {

  // calculate the norm of the rhs vector
  // TODO: save this from before
  double norm0 = b.Norm2();
  double dual_norm0 = b.dual().Norm2();
  
  // update the reduced-system RHS
  res_->dual() -= ceq; // VERY IMPORTANT; ceq is acctually x.dual()
  cout << "updating reduced-system RHS..." << endl;
  cout << "g_.size() = " << g_.size() << endl;
  cout << "iters_ = " << iters_ << endl;
  for (int k = 0; k < iters_+1; k++)
    g_(k) += InnerProd(res_->dual(), V_[k].dual());
  
  // compute the 2nd-order step
  cout <<"before solveLeastSquares..." << endl;
  solveLeastSquares(iters_+1, iters_, H_, g_, y_);
  res_->primal() = x.primal(); // save the objective gradient
  x = 0.0;
  cout <<"before computing x..." << endl;
  for (int k = 0; k < iters_; k++) {
    // x = x + y[k]*z[k]
    x.EqualsAXPlusBY(1.0, x, y_[k], Z_[k]);
  }
  double grad_dot_step = InnerProd(res_->primal(), x.primal());

  // compute residual
  ublas::vector_range<ublas::vector<double> >
      y_r(y_, ublas::range(0, iters_)),
      g_r(g_, ublas::range(0, iters_+1));
  ublas::matrix_range<ublas::matrix<double> >
      H_r(H_, ublas::range(0,iters_+1), ublas::range(0,iters_));
  cout <<"before res_red..." << endl;
  ublas::vector<double> res_red = g_r - ublas::prod(H_r, y_r);
  res_->primal() = 0.0;
  cout <<"before computing residual..." << endl;
  for (int k = 0; k < iters_+1; k++)
    res_->EqualsAXPlusBY(1.0, *res_, res_red(k), V_[k]);
  double beta = res_->Norm2();
  double gamma = res_->dual().Norm2();
  fout << "# second-order corrected step" << endl;
  writeKrylovHistory(fout, iters_+1, beta, norm0);  

  // update penalty parameter, if necessary  
  double pi = ptin.get<double>("pi");
  double ytHy = 0.0;
  UpdatePenaltyParameter(grad_dot_step, ytHy, ptin.get<double>("tau", 0.2),
                         dual_norm0, gamma, pi);  
  ptout.put<double>("pi", pi);
  ptout.put<double>("model_reduction", -grad_dot_step + pi*(dual_norm0-gamma));

  cout << "pi = " << pi << endl;
  cout << "grad_dot_step =" << grad_dot_step << endl;
  cout << "(dual_norm-gamma) = " << dual_norm0 - gamma << endl;
  cout << "norm2(y) = " << norm_2(y_r) << endl;
  //cout << "norm2(g) = " << res_->primal().Norm2() << endl;  
}
#endif
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
BPDSolver<Vec,PrimVec,DualVec>::BPDSolver() {
  maxiter_ = -1;
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void BPDSolver<Vec,PrimVec,DualVec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "BPDSolver::SubspaceSize(): "
         << "illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  maxiter_ = m;

  // Note: STL vectors are used for V_ and Z_ while boost::numeric::ublas vectors
  // are used for double based vectors and matrices; this choice was made
  // mostly to help emphasize that the objects in V_ and Z_ are fundamentally
  // different (they are some sort of vectors themselves).
  V_.reserve(maxiter_+1);
  Z_.reserve(maxiter_);
  g_.resize(maxiter_+1, 0.0);
  y_.resize(maxiter_, 0.0);
  H_.resize(maxiter_+1, maxiter_, 0.0);
  VtV_prim_.resize(maxiter_+1, maxiter_+1, 0.0);
  VtV_dual_.resize(maxiter_+1, maxiter_+1, 0.0);
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void BPDSolver<Vec,PrimVec,DualVec>::MemoryRequired(
    ptree& num_required) const {
  if (maxiter_ < 1) {
    cerr << "BPDSolver::MemoryRequired(): "
         << "SubspaceSize must be called first (maxiter_ is undefined)" << endl;
    throw(-1);
  }
  num_required.put("num_vec", 2*maxiter_ + 1 /* V and Z */ );
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void BPDSolver<Vec,PrimVec,DualVec>::Solve(
    const ptree & ptin, const Vec & b, Vec & x, 
    MatrixVectorProduct<Vec,Vec> & mat_vec, Preconditioner<Vec,Vec> & precond,
    ptree & ptout, ostream & fout) {
  V_.clear();
  Z_.clear();
  g_ = ublas::zero_vector<double>(g_.size());
  y_ = ublas::zero_vector<double>(y_.size());
  H_ = ublas::zero_matrix<double>(H_.size1(),H_.size2());
  VtV_prim_ = ublas::zero_matrix<double>(VtV_prim_.size1(),VtV_prim_.size2());
  VtV_dual_ = ublas::zero_matrix<double>(VtV_dual_.size1(),VtV_dual_.size2());
  iters_ = 0;
  
  // calculate the norm of the rhs vector
  double grad0 = b.primal().Norm2();
  double feas0 = b.dual().Norm2();
  double norm0 = sqrt(grad0*grad0 + feas0*feas0);

  // get the scalings
  double grad_scale = ptin.get<double>("grad_scale", 1.0);
  double feas_scale = ptin.get<double>("feas_scale", 1.0);

  // calculate the initial residual and compute its norm
  if (ptin.get<bool>("dynamic", false))
    mat_vec.set_product_tol(ptin.get<double>("tol")/
                            static_cast<double>(maxiter_));
  V_.push_back(b); // needed to initialize V_.[0], unfortunately
  V_[0].primal() *= grad_scale;
  V_[0].dual() *= feas_scale;

  double beta = V_[0].Norm2();
  double grad = V_[0].primal().Norm2();
  double feas = V_[0].dual().Norm2();

  // normalize residual to get v_{0}
  V_[0] /= beta;
  VtV_prim_(0,0) = InnerProd(V_[0].primal(), V_[0].primal());
  VtV_dual_(0,0) = InnerProd(V_[0].dual(), V_[0].dual());
  
  // initialize the RHS of the reduced system
  g_(0) = beta;

  // output header information including initial residual
  int i = 0;
  WriteHeader(fout, ptin.get<double>("tol"), norm0, grad0, feas0);
  double res_norm = norm0;
  WriteHistory(fout, i, res_norm/norm0, grad/(grad_scale),
               feas/(feas_scale));

  // loop over all search directions
  bool lin_depend = false;
  for (i = 0; i < maxiter_; i++) {
    iters_++;

    // precondition the Vec V_[i] and store result in Z_[i]
    Z_.push_back(b);
    precond(V_[i], Z_[i]);
    
    // add to Krylov subspace
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
                              (beta*static_cast<double>(maxiter_)));
    V_.push_back(Z_[i]);
    Z_[i].primal() *= grad_scale;
    Z_[i].dual() *= feas_scale;
    mat_vec(Z_[i], V_[i+1]);
    Z_[i].primal() /= grad_scale;
    Z_[i].dual() /= feas_scale;
    V_[i+1].primal() *= grad_scale;
    V_[i+1].dual() *= feas_scale;

    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H_, V_);
    } catch (string err_msg) {
      cerr << "BPD: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

    // compute new row and column of VtV_* matrices
    for (int k = 0; k <= i; k++) {
      VtV_prim_(k,i+1) = InnerProd(V_[k].primal(), V_[i+1].primal());
      VtV_prim_(i+1,k) = VtV_prim_(k,i+1);
      VtV_dual_(k,i+1) = InnerProd(V_[k].dual(), V_[i+1].dual());
      VtV_dual_(i+1,k) = VtV_dual_(k,i+1);
    }
    VtV_prim_(i+1,i+1) = InnerProd(V_[i+1].primal(), V_[i+1].primal());
    VtV_dual_(i+1,i+1) = InnerProd(V_[i+1].dual(), V_[i+1].dual());
    
    // solve the reduced problem
    solveLeastSquares(iters_+1, iters_, H_, g_, y_);

    // these ranges help improve readability below
    ublas::vector_range<ublas::vector<double> >
        y_r(y_, ublas::range(0, iters_)),
        g_r(g_, ublas::range(0, iters_+1));
    ublas::matrix_range<const ublas::matrix<double> >
        VtV_prim_r(VtV_prim_, ublas::range(0,iters_+1), ublas::range(0,iters_+1)),
        VtV_dual_r(VtV_dual_, ublas::range(0,iters_+1), ublas::range(0,iters_+1)),
        H_r(H_, ublas::range(0,iters_+1), ublas::range(0,iters_));
    
    // compute the residual norms
    ublas::vector<double> res_red = ublas::prod(H_r, y_r) - g_r;
    beta = norm_2(res_red);
    grad = ublas::inner_prod(res_red, ublas::prod(VtV_prim_r, res_red));
    grad = sqrt(std::max(0.0, grad));
    feas = ublas::inner_prod(res_red, ublas::prod(VtV_dual_r, res_red));
    feas = sqrt(std::max(0.0, feas));
    
    // output history
    res_norm = pow(grad/grad_scale,2.0) + pow(feas/feas_scale,2.0);
    res_norm = sqrt(std::max(0.0, res_norm));
    WriteHistory(fout, i+1, res_norm/norm0, grad/(grad_scale),
                 feas/(feas_scale));
    
    // check for convergence
    if ( (res_norm < norm0*ptin.get<double>("tol")) &&
         (grad < ptin.get<double>("grad_tol")) &&
         (feas < ptin.get<double>("feas_tol")) )
      break;
  }  
  ptout.put<int>("iters", iters_);
  ptout.put<double>("res", res_norm);

  // compute the full-space solution
  x.primal() = 0.0;
  x.dual() = 0.0;
  for (int k = 0; k < iters_; k++) {
    x.primal().EqualsAXPlusBY(1.0, x.primal(), grad_scale*y_(k), Z_[k].primal());
    x.dual().EqualsAXPlusBY(1.0, x.dual(), feas_scale*y_(k), Z_[k].dual());
  }
  
  if (ptin.get<bool>("check", false)) {
    // recalculate explicilty and check final residual
    if (ptin.get<bool>("dynamic", false))
      mat_vec.set_product_tol(0.1*ptin.get<double>("tol")/
                              static_cast<double>(maxiter_));
    mat_vec(x, V_[0]);
    // V_[0] = b - V_[0] = b - A*x
    V_[0].EqualsAXPlusBY(1.0, b, -1.0, V_[0]);
    double true_res = V_[0].Norm2();
    double true_grad = V_[0].primal().Norm2();
    double true_feas = V_[0].dual().Norm2();
    fout << "# BPD final (true) residual : |res|/|res0| = "
         << true_res/norm0 << endl;
    fout << "# BPD final (true) optimality : |grad| = "
         << true_grad << endl;
    fout << "# BPD final (true) feas.  : |feas| = "
         << true_feas << endl;
    ptout.put<double>("res", true_res);
    if (fabs(true_res - res_norm) > 0.01*ptin.get<double>("tol")*norm0) {
      fout << "# WARNING in BPDSolver: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = "
           << (true_res - res_norm)/norm0 << endl;
    }
  }
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void BPDSolver<Vec,PrimVec,DualVec>::WriteHeader(
    ostream& os, const double& tol, const double& res0,
    const double& grad0, const double& feas0) {
#if 0
  if (!(os.good())) {
    cerr << "BPDSolver::WriteHeader(): "
	 << "ostream is not good for i/o operations." << endl;
    throw(-1);
  }
#endif
  os << "# BPD convergence history" << endl;
  os << boost::format(
      "# residual tolerance target %|30t| = %|-10.6|\n") % tol;
  os << boost::format(
      "# initial residual norm %|30t| = %|-10.6|\n") % res0;
  os << boost::format(
      "# initial optimality %|30t| = %|-10.6|\n") % grad0;
  os << boost::format(
      "# initial feasibility %|30t| = %|-10.6|\n") % feas0;
  os << boost::format(string("%|6| %|8t| %|-12| %|20t| %|-12|")+
                      string(" %|32t| %|-12|\n"))
      % "# iter" % "rel. res." % "abs. opt." % "abs. feas.";
  os.flush();
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void BPDSolver<Vec,PrimVec,DualVec>::WriteHistory(
    ostream& os, const int& iter, const double& res, const double& grad,
    const double& feas) {
#if 0
  if (!(os.good())) {
    cerr << "BPDSolver::WriteHistory(): "
	 << "ostream is not good for i/o operations." << endl;
    throw(-1);
  }
#endif
  os << boost::format(string("%|6| %|8t| %|-12.6| %|20t| %|-12.6|")+
                      string(" %|32t| %|-12.6|\n"))
      % iter % res % grad % feas;
  os.flush();
}
// ==============================================================================
template <class Vec>
void modGramSchmidt(int i, ublas::matrix<double> & Hsbg, vector<Vec> & w) {
  string err_msg ("modGramSchmidt failed: ");
  const double reorth = 0.98;

  // get the norm of the vector being orthogonalized, and find the
  // threshold for re-orthogonalization
  double nrm = InnerProd(w[i+1], w[i+1]);
  double thr = nrm*reorth;
  if (fabs(nrm) <= kEpsilon) {
    // the norm of w[i+1] is effectively zero; it is linearly dependent
    throw(true);
  } else if (nrm < -kEpsilon) {
    // the norm of w[i+1] < 0.0
    err_msg.append("InnerProd(w[i+1], w[i+1]) = ");
    err_msg.append(boost::lexical_cast<std::string>(nrm));
    err_msg.append(" < 0.0");
    throw(err_msg);
  } else if (nrm != nrm) {
    // this is intended to catch if nrm = NaN, but some optimizations
    // may mess it up (according to posts on stackoverflow.com)
    err_msg.append("w[i+1] = NaN");
    throw(err_msg);
  }

  if (i < 0) {
    // just normalize and exit
    w[i+1] /= sqrt(nrm);
    return;
  }
  
  // begin main Gram-Schmidt loop
  for (int k = 0; k < i+1; k++) {
    //for (int k = i; k >=0; k--) {
    double prod = InnerProd(w[i+1], w[k]);
    Hsbg(k, i) = prod;
    // w[i+1] = w[i+1] - prod*w[k]
    w[i+1].EqualsAXPlusBY(1.0, w[i+1], -prod, w[k]);

    // check if reorthogonalization is necessary
    if (prod*prod > thr) {
      prod = InnerProd(w[i+1], w[k]);
      Hsbg(k, i) += prod;
      // w[i+1] = w[i+1] - prod*w[k]
      w[i+1].EqualsAXPlusBY(1.0, w[i+1], -prod, w[k]);
    }

    // update the norm and check its size
    nrm -= Hsbg(k, i)*Hsbg(k, i);
    if (nrm < 0.0) nrm = 0.0;
    thr = nrm*reorth;
  }

  // test the resulting vector
  nrm = w[i+1].Norm2();
  Hsbg(i+1, i) = nrm;
  if (nrm <= 0.0) {
    // w[i+1] is a linear combination of the w[0:i]
    //err_msg.append("w[i+1] linearly dependent with w[0:i]");
    throw(true);
    //throw(err_msg);
  } else {
    // scale the resulting vector
    w[i+1] /= nrm;
  }
}
// ==============================================================================
template <class Vec>
void modGramSchmidt(int i, vector<Vec> & w) {
  string err_msg ("modGramSchmidt failed: ");
  const double reorth = 0.98;

  // get the norm of the vector being orthogonalized, and find the
  // threshold for re-orthogonalization
  double nrm = InnerProd(w[i+1], w[i+1]);
  double thr = nrm*reorth;
  if (nrm <= 0.0) {
    // the norm of w[i+1] < 0.0
    err_msg.append("InnerProd(w[i+1], w[i+1]) < 0.0");
    throw(err_msg);
  } else if (nrm != nrm) {
    // this is intended to catch if nrm = NaN, but some optimizations
    // may mess it up (according to posts on stackoverflow.com)
    err_msg.append("w[i+1] = NaN");
    throw(err_msg);
  }

  if (i < 0) {
    // just normalize and exit
    w[i+1] /= sqrt(nrm);
    return;
  }
  
  // begin main Gram-Schmidt loop
  for (int k = 0; k < i+1; k++) {
  //for (int k = i; k >=0; k--) {
    double prod = InnerProd(w[i+1], w[k]);
    double Hsbg = prod;
    // w[i+1] = w[i+1] - prod*w[k]
    w[i+1].EqualsAXPlusBY(1.0, w[i+1], -prod, w[k]);

    // check if reorthogonalization is necessary
    if (prod*prod > thr) {
      prod = InnerProd(w[i+1], w[k]);
      Hsbg += prod;
      // w[i+1] = w[i+1] - prod*w[k]
      w[i+1].EqualsAXPlusBY(1.0, w[i+1], -prod, w[k]);
    }

    // update the norm and check its size
    nrm -= Hsbg*Hsbg;
    if (nrm < 0.0) nrm = 0.0;
    thr = nrm*reorth;
  }

  // test the resulting vector
  nrm = w[i+1].Norm2();
  if (nrm <= 0.0) {
    // w[i+1] is a linear combination of the w[0:i]
    //err_msg.append("w[i+1] linearly dependent with w[0:i]");
    throw(true);
    //throw(err_msg);
  } else {
    // scale the resulting vector
    w[i+1] /= nrm;
  }
}
// ==============================================================================
template <class Vec>
double Lanczos(int m, double tol, MatrixVectorProduct<Vec> & mat_vec,
               int & iters, ostream & fout = cout) {
  // check the subspace size
  if (m < 1) {
    cerr << "Lanczos: illegal value for max. iterations, m = "
         << m << endl;
    throw(-1);
  }

  // v and w are as in Algorithm 6.5 in Saad's book on eigenvalue comp.
  // and z corresponds to v_{j-1}
  Vec v, w, z;
  ublas::matrix<double> H(m, m, 0.0);
  ublas::vector<double> eig(m);
  
  v = 1.0;
  z = 0.0;
  v /= v.Norm2();

  double max_eig = 0.0;
  double old_eig = 0.0;
  iters = 0;
  int i;
  for (i = 0; i < m; i++) {
    mat_vec(v, w);
    if (i > 0) w.EqualsAXPlusBY(1.0, w, -H(i-1,i), z);
    H(i,i) = InnerProd(w, v);

    // compute the eigenvalues of the tridiagonal matrix H, and find current
    // estimate for max |eig(A)|    
    eigenvalues(i+1, H, eig);
    old_eig = max_eig;
    max_eig = 0.0;    
    for (int j = 0; j < i+1; j++) {
      max_eig = std::max(max_eig, fabs(eig(j)));
    }
    double diff_eig = fabs(max_eig - old_eig);
    writeKrylovHistory(fout, i, diff_eig, max_eig);

    // TEMP: estimate inertia
    int num_pos = 0, num_neg = 0, num_zero = 0;
    for (int j = 0; j < i+1; j++) {
      if (eig(j) > 1e-100)
        num_pos++;
      else if (eig(j) < -1e100)
        num_neg++;
      else
        num_zero++;
    }
    cout << "matrix inertia = (" << num_pos << "," << num_neg << ","
         << num_zero << ")\n";    
    //if (diff_eig < tol*max_eig) return max_eig;

    // prevents accessing nonexisting elements of H
    if (i == m-1) break;

    z = v;
    v *= H(i,i);
    w -= v;
    H(i,i+1) = w.Norm2();
    H(i+1,i) = H(i,i+1);
    v = w;
    v /= H(i,i+1);
  }
  // if we get here, we failed to find an estimate to within the desired
  // tolerance; warn user and return max_eig estimate
  fout << "Lanczos: WARNING, did not find eigenvalue estimate within given "
       << "number of iterations" << endl;
  return max_eig;
}

// ==============================================================================
template <class Vec>
void GMRES(int m, double tol, const Vec & b, Vec & x,
           MatrixVectorProduct<Vec> & mat_vec,
           Preconditioner<Vec> & precond, int & iters, 
           ostream & fout = cout, const bool & check_res = true,
           const bool & dynamic = false) {
  // check the subspace size
  if (m < 1) {
    cerr << "GMRES: illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }

  // define various vectors and arrays

  // Note: elements in w and z are initialized to x to avoid creating
  // a temporary Vec object from the copy constructor

  // Note: STL vectors are used for w while boost::numeric::ublas vectors are
  // used for double based vectors and matrices; this choice was made mostly to
  // help emphasize that the objects in w are fundamentally different (they are
  // some sort of vectors themselves).
  vector<Vec> w(m+1, x);
  Vec z(x);
  ublas::vector<double> g(m+1, 0.0);
  ublas::vector<double> sn(m+1, 0.0);
  ublas::vector<double> cs(m+1, 0.0);
  ublas::vector<double> y(m, 0.0);
  ublas::matrix<double> H(m+1, m, 0.0);
  iters = 0;

  // calculate the norm of the rhs vector
  double norm0 = b.Norm2();

  // calculate the initial residual (actually the negative residual)
  // and compute its norm
  if (dynamic) mat_vec.set_product_tol(0.5*tol);
  mat_vec(x, w[0]);
  w[0] -= b;

  double beta = w[0].Norm2();
  if ( (beta < tol*norm0) || (beta < kEpsilon) ) {
    // system is already solved
    fout << "GMRES system solved by initial guess." << endl;
    return;
  }

  // normalize residual to get w_{0} (the negative sign is because w[0]
  // holds the negative residual, as mentioned above)
  w[0] /= -beta;

  // initialize the RHS of the reduced system
  g(0) = beta;

  // output header information including initial residual
  int i = 0;
  writeKrylovHeader(fout, "GMRES", tol, beta);
  writeKrylovHistory(fout, i, beta, norm0);

  // loop over all search directions
  bool lin_depend = false;
  for (i = 0; i < m; i++) {
    // check if solution has converged; also, if we have a w vector
    // that is linearly dependent, check that we have converged
    if ( (lin_depend) && (beta > tol*norm0) ) {
      cerr << "GMRES: Arnoldi process breakdown: "
           << "H(" << i+1 << "," << i << ") = " << H(i+1,i)
           << ", however ||res|| = " << beta << endl;
      throw(-1);
    } else if (beta < tol*norm0) break;
    iters++;

    // precondition the Vec w[i] and store result in z
    precond(w[i], z);

    // add to Krylov subspace
    if (dynamic) mat_vec.set_product_tol(0.5*norm0*tol/beta);
    mat_vec(z, w[i+1]);

    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H, w);
    } catch (string err_msg) {
      cerr << "GMRES: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

    // apply old Givens rotations to new column of the Hessenberg matrix then
    // generate the new Givens rotation matrix and apply it to the last two
    // elements of H(i,:) and g
    for (int k = 0; k < i; k++)
      applyGivens(sn(k), cs(k), H(k,i), H(k+1,i));
    generateGivens(H(i,i), H(i+1,i), sn(i), cs(i));
    applyGivens(sn(i), cs(i), g(i), g(i+1));

    // set L2 norm of residual and output the relative residual if necessary
    beta = fabs(g(i+1));
    writeKrylovHistory(fout, i+1, beta, norm0);
  }

  // solve the least-squares system and update solution
  solveReducedHessenberg(i, H, g, y);
  z = 0.0;
  for (int k = 0; k < i; k++) {
    // x = x + y[k]*w[k]
    z.EqualsAXPlusBY(1.0, z, y[k], w[k]);
  }
  precond(z, w[0]);
  x += w[0];

  if (check_res) {
    // recalculate explicilty and check final residual
    if (dynamic) mat_vec.set_product_tol(0.1*tol);
    mat_vec(x, w[0]);
    // w[0] = b - w[0] = b - A*x
    w[0].EqualsAXPlusBY(1.0, b, -1.0, w[0]);

    double res = w[0].Norm2();
    fout << "# GMRES final (true) residual : |res|/|res0| = "
         << res/norm0 << endl;
    
    if (fabs(res - beta) > 0.01*tol*norm0) {
      fout << "# WARNING in GMRES: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = " << (res - beta)/norm0 << endl;
    }
  }
}
// ==============================================================================
template <class Vec>
void FGMRES(int m, double tol, const Vec & b, Vec & x,
	    MatrixVectorProduct<Vec> & mat_vec,
	    Preconditioner<Vec> & precond, int & iters, 
            ostream & fout = cout, const bool & check_res = true,
            const bool & dynamic = false, const boost::scoped_ptr<double> & res
            = boost::scoped_ptr<double>(), const boost::scoped_ptr<double> & sig
            = boost::scoped_ptr<double>()) {
  // check the subspace size
  if (m < 1) {
    cerr << "FGMRES: illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }

  // define various vectors and arrays

  // Note: elements in w and z are initialized to x to avoid creating
  // a temporary Vec object from the copy constructor

  // Note: STL vectors are used for w and z while boost::numeric::ublas vectors
  // are used for double based vectors and matrices; this choice was made
  // mostly to help emphasize that the objects in w and z are fundamentally
  // different (they are some sort of vectors themselves).
  vector<Vec> w(m+1, x);
  vector<Vec> z(m, x);
  ublas::vector<double> g(m+1, 0.0);
  ublas::vector<double> sn(m+1, 0.0);
  ublas::vector<double> cs(m+1, 0.0);
  ublas::vector<double> y(m, 0.0);
  ublas::matrix<double> H(m+1, m, 0.0);
  iters = 0;

  // calculate the norm of the rhs vector
  double norm0 = b.Norm2();

  // calculate the initial residual (actually the negative residual)
  // and compute its norm
  if (dynamic) mat_vec.set_product_tol(tol/static_cast<double>(m));
  mat_vec(x, w[0]);
  w[0] -= b;

  double beta = w[0].Norm2();
  if ( (beta < tol*norm0) || (beta < kEpsilon) ) {
    // system is already solved
    fout << "FGMRES system solved by initial guess." << endl;
    if (res) *res = beta; 
    return;
  }

  // normalize residual to get w_{0} (the negative sign is because w[0]
  // holds the negative residual, as mentioned above)
  w[0] /= -beta;

  // initialize the RHS of the reduced system
  g(0) = beta;

  // output header information including initial residual
  int i = 0;
  writeKrylovHeader(fout, "FGMRES", tol, beta);
  writeKrylovHistory(fout, i, beta, norm0);

  // loop over all search directions
  bool lin_depend = false;
  for (i = 0; i < m; i++) {
    // check if solution has converged; also, if we have a w vector
    // that is linearly dependent, check that we have converged
    if ( (lin_depend) && (beta > tol*norm0) ) {
      cerr << "FGMRES: Arnoldi process breakdown: "
           << "H(" << i+1 << "," << i << ") = " << H(i+1,i)
           << ", however ||res|| = " << beta << endl;
      throw(-1);
    } else if (beta < tol*norm0) break;
    iters++;

    // precondition the Vec w[i] and store result in z[i]
    precond(w[i], z[i]);

    // add to Krylov subspace
    if (dynamic) mat_vec.set_product_tol(
            tol*norm0/(beta*static_cast<double>(m)));
    mat_vec(z[i], w[i+1]);

    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H, w);
    } catch (string err_msg) {
      cerr << "FGMRES: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

    // apply old Givens rotations to new column of the Hessenberg matrix
    // then generate the new Givens rotation matrix and apply it to
    // the last two elements of H(i,:) and g
    for (int k = 0; k < i; k++)
      applyGivens(sn(k), cs(k), H(k,i), H(k+1,i));
    generateGivens(H(i,i), H(i+1,i), sn(i), cs(i));
    applyGivens(sn(i), cs(i), g(i), g(i+1));

    // set L2 norm of residual and output the relative residual if necessary
    beta = fabs(g(i+1));
    writeKrylovHistory(fout, i+1, beta, norm0);
  }

  // solve the least-squares system and update solution
  solveReducedHessenberg(i, H, g, y);
  for (int k = 0; k < i; k++) {
    // x = x + y[k]*z[k]
    x.EqualsAXPlusBY(1.0, x, y[k], z[k]);
  }

  if (res) *res = beta;

  if (sig) {
    // compute smallest singlular value of H
    ublas::vector<double> Sigma, P, UT;
    computeSVD(i+1, i, H, Sigma, P, UT);
    cout << "[ j singular value of H = " << Sigma(i-1) << endl;
  }
  
  if (check_res) {
    // recalculate explicilty and check final residual
    if (dynamic) mat_vec.set_product_tol(0.1*tol/static_cast<double>(m));
    mat_vec(x, w[0]);
    // w[0] = b - w[0] = b - A*x
    w[0].EqualsAXPlusBY(1.0, b, -1.0, w[0]);

    double true_res = w[0].Norm2();
    fout << "# FGMRES final (true) residual : |res|/|res0| = "
         << true_res/norm0 << endl;
    if (res) *res = true_res;
    if (fabs(true_res - beta) > 0.01*tol*norm0) {
      fout << "# WARNING in FGMRES: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = " << (true_res - beta)/norm0 << endl;
    }
  }
}
// ==============================================================================
template <class Vec>
void FFOM(int m, double tol, const Vec & b, Vec & x,
          MatrixVectorProduct<Vec> & mat_vec,
          Preconditioner<Vec> & precond, int & iters, 
          ostream & fout = cout, const bool & check_res = true,
          const bool & dynamic = false) {
  // check the subspace size
  if (m < 1) {
    cerr << "FFOM: illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }

  // define various vectors and arrays

  // Note: elements in w and z are initialized to x to avoid creating
  // a temporary Vec object from the copy constructor

  // Note: STL vectors are used for w and z while
  // boost::numeric::ublas vectors are used for double based vectors
  // and matrices; this choice was made mostly to help emphasize that
  // the objects in w and z are fundamentally different (they are
  // some sort of vectors themselves).
  vector<Vec> w(m+1, x);
  vector<Vec> z(m, x);
  ublas::vector<double> g(m+1, 0.0);
  ublas::vector<double> sn(m+1, 0.0);
  ublas::vector<double> cs(m+1, 0.0);
  ublas::vector<double> y(m, 0.0);
  ublas::vector<double> y_old(m, 0.0);
  ublas::vector<double> bZ(m, 0.0);
  ublas::matrix<double> H(m+1, m, 0.0);
  iters = 0;

  // calculate the norm of the rhs vector
  double norm0 = b.Norm2();

  // calculate the initial residual (actually the negative residual)
  // and compute its norm
  if (dynamic) mat_vec.set_product_tol(tol/static_cast<double>(m));
  mat_vec(x, w[0]);
  w[0] -= b;

  double beta = w[0].Norm2();
  if ( (beta < tol*norm0) || (beta < kEpsilon) ) {
    // system is already solved
    fout << "FFOM system solved by initial guess." << endl;
    return;
  }

  // normalize residual to get w_{0} (the negative sign is because w[0]
  // holds the negative residual, as mentioned above)
  w[0] /= -beta;

  // initialize the RHS of the reduced system
  g(0) = beta;

  // output header information including initial residual
  int i = 0;
  writeKrylovHeader(fout, "FFOM", tol, beta);
  writeKrylovHistory(fout, i, beta, norm0);

  // loop over all search directions
  bool lin_depend = false;
  double beta_old = norm0; // not needed anymore?
  for (i = 0; i < m; i++) {
    // check if solution has converged
    if (beta < tol*norm0) break;
    iters++;

    // precondition the Vec w[i] and store result in z[i]
    precond(w[i], z[i]);

    // compute b^{T} z[i] for descent direction check later
    bZ(i) = InnerProd(b, z[i]);

    // add to Krylov subspace
    if (dynamic) mat_vec.set_product_tol(
            tol*norm0/(beta*static_cast<double>(m)));
    mat_vec(z[i], w[i+1]);

    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H, w);
    } catch (string err_msg) {
      cerr << "FFOM: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

    // apply old Givens rotations to new column of the Hessenberg matrix
    for (int k = 0; k < i; k++)
      applyGivens(sn(k), cs(k), H(k,i), H(k+1,i));

    // Check if the reduced system is singular; if not, solve the reduced
    // square system and compute the new residual norm
    if (triMatixInvertible(i+1, H)) {
      solveReducedHessenberg(i+1, H, g, y);
      beta_old = beta;
      beta = fabs(y(i))*H(i+1,i);
#if 0
      // check for ascent direction
      double b_dot_x = 0.0;
      for (int k = 0; k < i+1; k++)
        b_dot_x += bZ(k)*y(k);
      if (b_dot_x <= 0.0) { 
        // -x is an ascent direction, so exit
        cout << "FFOM: found an ascent direction; b_dot_x = "
             << b_dot_x  << endl;
        writeKrylovHistory(fout, i+1, beta, norm0);
        i++;
        break;
      }
#endif
    }
    writeKrylovHistory(fout, i+1, beta, norm0);

    // if we have a w vector that is linearly dependent, check that we have
    // converged
    if ( (lin_depend) && (beta > tol*norm0) ) {
      cerr << "FFOM: Arnoldi process breakdown: "
           << "H(" << i+1 << "," << i << ") = " << H(i+1,i)
           << ", however ||res|| = " << beta << endl;
      throw(-1);
    } 
    
    // Generate the new Givens rotation matrix and apply it to
    // the last two elements of H(i,:) and g
    generateGivens(H(i,i), H(i+1,i), sn(i), cs(i));
    applyGivens(sn(i), cs(i), g(i), g(i+1));
  }

  // compute solution
  for (int k = 0; k < i; k++) {
    // x = x + y[k]*z[k]
    x.EqualsAXPlusBY(1.0, x, y[k], z[k]);
  }

  if (check_res) {
    // recalculate explicilty and check final residual
    if (dynamic) mat_vec.set_product_tol(0.1*tol/static_cast<double>(m));
    mat_vec(x, w[0]);
    // w[0] = b - w[0] = b - A*x
    w[0].EqualsAXPlusBY(1.0, b, -1.0, w[0]);

    double res = w[0].Norm2();
    fout << "# FFOM final (true) residual : |res|/|res0| = "
         << res/norm0 << endl;
    
    if (fabs(res - beta) > 0.01*tol*norm0) {
      fout << "# WARNING in FFOM: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = " << (res - beta)/norm0 << endl;
    }
  }
}
// ==============================================================================
template <class Vec>
void SteihaugCG(int m, double tol, double radius, const Vec & b, Vec & x,
                MatrixVectorProduct<Vec> & mat_vec,
                Preconditioner<Vec> & precond, int & iters,
                double & pred, bool & active,
                ostream & fout = cout, const bool & check_res = true,
                const bool & dynamic = false) {
  // check the subspace size
  if (m < 1) {
    cerr << "SteihaugCG: illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  if (radius <= 0.0) {
    cerr << "SteihaugCG: trust-region radius must be positive, radius = "
         << radius << endl;
    throw(-1);
  }
  
  // define various vectors and scalars
  Vec r(b);
  x = 0.0;
  double alpha;  
  double x_norm2 = 0.0;
#if 0
  vector<Vec> w(m+1, x);
  vector<Vec> z(m, x);
  ublas::vector<double> g(m+1, 0.0);
  ublas::vector<double> sn(m+1, 0.0);
  ublas::vector<double> cs(m+1, 0.0);
  ublas::vector<double> y(m, 0.0);
  ublas::matrix<double> H(m+1, m, 0.0);
#endif
  iters = 0;

  // calculate the norm of the rhs vector
  double norm0 = r.Norm2();

  if (norm0 < kEpsilon) {
    // system is already solved
    fout << "SteihaugCG: system solved by initial guess." << endl;
    return;
  }
  double res_norm2 = norm0;
  Vec z(r);
  precond(r, z);
  double rdotz = InnerProd(r, z);
  Vec p(z);
  Vec Ap(p);

  // output header information including initial residual
  int i = 0;
  //writeKrylovHeader(fout, "SteihaugCG", tol, res_norm2);
  //writeKrylovHistory(fout, i, res_norm2, norm0);

  // loop over search directions
  for (i = 0; i < m; i++) {
    iters++;

    if (dynamic) mat_vec.set_product_tol(
            tol*norm0/(res_norm2*static_cast<double>(m)));
    mat_vec(p, Ap);
    alpha = InnerProd(p, Ap);
    if (alpha <= 0) {
      // direction of nonpositive curvature detected
      double xp = InnerProd(x, p);
      double x2 = x_norm2*x_norm2;
      double p2 = InnerProd(p, p);
      double tau = (-xp + sqrt(xp*xp - p2*(x2 - radius*radius)))/p2;
      x.EqualsAXPlusBY(1.0, x, tau, p);
      r.EqualsAXPlusBY(1.0, r, -tau, Ap);
      res_norm2 = r.Norm2();
      if (iters == 1) {      
        writeKrylovHeader(fout, "SteihaugCG", tol, res_norm2);
        norm0 = res_norm2;
      }
      writeKrylovHistory(fout, iters, res_norm2, norm0);
      fout << "# direction of nonpositive curvature detected" << endl;
      active = true;
      break;
    }

    alpha = rdotz/alpha;
    x.EqualsAXPlusBY(1.0, x, alpha, p);
    x_norm2 = x.Norm2();
    if (x_norm2 >= radius) {
      x.EqualsAXPlusBY(1.0, x, -alpha, p);
      double xp = InnerProd(x, p);
      double x2 = InnerProd(x, x);
      double p2 = InnerProd(p, p);
      double tau = (-xp + sqrt(xp*xp - p2*(x2 - radius*radius)))/p2;
      x.EqualsAXPlusBY(1.0, x, tau, p);
      r.EqualsAXPlusBY(1.0, r, -tau, Ap);
      res_norm2 = r.Norm2();
      if (iters == 1) {      
        writeKrylovHeader(fout, "SteihaugCG", tol, res_norm2);
        norm0 = res_norm2;
      }
      writeKrylovHistory(fout, iters, res_norm2, norm0);
      fout << "# trust-region boundary encountered" << endl;
      active = true;
      break;
    }

    // compare residual and check for convergence
    r.EqualsAXPlusBY(1.0, r, -alpha, Ap);
    res_norm2 = r.Norm2();

    if (iters == 1) {      
      writeKrylovHeader(fout, "SteihaugCG", tol, res_norm2);
      norm0 = res_norm2;
    }
    writeKrylovHistory(fout, iters, res_norm2, norm0);
    if (res_norm2 < norm0*tol) {
      break;
    }

    precond(r, z);
    double beta = 1.0/rdotz;
    rdotz = InnerProd(r, z);
    beta *= rdotz;
    p.EqualsAXPlusBY(1.0, z, beta, p);    
  }

  // compute the predicted reduction in the objective
  // Note: -Ax = r - b --> x^T b - 0.5 x^T A x = 0.5*x^T(b + r)
  r.EqualsAXPlusBY(1.0, r, 1.0, b);
  pred = 0.5*InnerProd(x, r);
  
  if (check_res) {
    // check the residual norm
    if (dynamic) mat_vec.set_product_tol(0.1*tol/static_cast<double>(m));
    mat_vec(x, r);
    r.EqualsAXPlusBY(1.0, b, -1.0, r);
    double res = r.Norm2();
    fout << "# SteihaugCG final (true) residual : |res|/|res0| = "
         << res/norm0 << endl;    
    if (fabs(res - res_norm2) > 0.01*tol*norm0) {
      fout << "# WARNING in SteihaugCG: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = " << (res - res_norm2)/norm0 << endl;
    }
    // check that solution satisfies the trust region
    x_norm2 = x.Norm2();
    if (x_norm2 - radius > 1e-6) {
      cerr << "SteihaugCG: solution is outside trust region."
           << endl;
      throw(-1);
    }
  }
}
// ==============================================================================
template <class Vec>
void SteihaugFOM(int m, double tol, double radius, const Vec & b, Vec & x,
                 MatrixVectorProduct<Vec> & mat_vec,
                 Preconditioner<Vec> & precond, int & iters, double & pred,
                 bool & active, ostream & fout = cout,
                 const bool & check_res = true, const bool & dynamic = false) {
  // check the subspace size
  if (m < 1) {
    cerr << "SteihaugFOM: illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  if (radius <= 0.0) {
    cerr << "SteihaugFOM: trust-region radius must be positive, radius = "
         << radius << endl;
    throw(-1);
  }

  cerr << "SteihaugFOM: not implememnted yet." << endl;
  throw(-1);
  
  // define various vectors and arrays

  // Note: elements in w and z are initialized to x to avoid creating
  // a temporary Vec object from the copy constructor

  // Note: STL vectors are used for w while boost::numeric::ublas vectors are
  // used for double based vectors and matrices; this choice was made mostly to
  // help emphasize that the objects in w are fundamentally different (they are
  // some sort of vectors themselves).
  vector<Vec> w(m+1, x);
  Vec z(x);
  ublas::vector<double> g(m+1, 0.0);
  ublas::vector<double> sn(m+1, 0.0);
  ublas::vector<double> cs(m+1, 0.0);
  ublas::vector<double> y(m, 0.0);
  ublas::vector<double> y_old(m, 0.0);
  ublas::matrix<double> H(m+1, m, 0.0);
  iters = 0;

  // calculate the norm of the rhs vector
  double norm0 = b.Norm2();

  // calculate the initial residual norm
  w[0] = b;
  double beta = w[0].Norm2();
  if ( (beta < tol*norm0) || (beta < kEpsilon) ) {
    // system is already solved or at a Stationary point
    fout << "SteihaugFOM system solved by initial guess." << endl;
    return;
  }

  // normalize residual to get w_{0}
  w[0] /= beta;

  // initialize the RHS of the reduced system
  g(0) = beta;

  // output header information including initial residual
  int i = 0;
  writeKrylovHeader(fout, "SteihaugFOM", tol, beta);
  writeKrylovHistory(fout, i, beta, norm0);

  // loop over all search directions
  bool lin_depend = false;
  for (i = 0; i < m; i++) {
    // check if solution has converged
    if (beta < tol*norm0) break;
    iters++;

    // precondition the Vec w[i] and store result in z
    precond(w[i], z);

    // add to Krylov subspace
    if (dynamic) mat_vec.set_product_tol(
            tol*norm0/(beta*static_cast<double>(m)));
    mat_vec(z, w[i+1]);

    // need to predict negative curvature directions
    
    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H, w);
    } catch (string err_msg) {
      cerr << "SteihaugFOM: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }

    // apply old Givens rotations to new column of the Hessenberg matrix
    for (int k = 0; k < i; k++)
      applyGivens(sn(k), cs(k), H(k,i), H(k+1,i));
    
    // Check if the reduced system is singular; if not, solve the reduced
    // square system and compute the new residual norm
    if (triMatixInvertible(i+1, H)) {
      y_old = y;
      solveReducedHessenberg(i+1, H, g, y);
      beta = fabs(y(i))*H(i+1,i);
      // check norm of solution (which is equal to ||y||)
      double y_norm2 = norm_2(y);
      if (y_norm2 > radius) {
        double y_i = y(i);
        for (int k = 0; k < i; k++) y(k) = y_old(k); 
        y(i) = norm_2(y_old);
        y(i) = sqrt(radius*radius - y(i)*y(i));
        beta = 0.0;
        for (int k = 0; k <= i+1; k++) beta += H(k,i)*H(k,i);
        beta = fabs(y(i) - y_i)*sqrt(beta);
        writeKrylovHistory(fout, i+1, beta, norm0);
        break;
      }
    }
    writeKrylovHistory(fout, i+1, beta, norm0);

    // if we have a w vector that is linearly dependent, check that we have
    // converged
    if ( (lin_depend) && (beta > tol*norm0) ) {
      cerr << "SteihaugFOM Arnoldi process breakdown: "
           << "H(" << i+1 << "," << i << ") = " << H(i+1,i)
           << ", however ||res|| = " << beta << endl;
      throw(-1);
    } 
    
    // generate the new Givens rotation matrix and apply it to the last two
    // elements of H(i,:) and g    
    generateGivens(H(i,i), H(i+1,i), sn(i), cs(i));
    applyGivens(sn(i), cs(i), g(i), g(i+1));
  }

  // compute solution
  z = 0.0;
  for (int k = 0; k < i; k++) {
    // x = x + y[k]*w[k]
    z.EqualsAXPlusBY(1.0, z, y[k], w[k]);
  }
  precond(z, w[0]);
  x = w[0];

  // need to compute predicted decrease in objective

  if (check_res) {
    // recalculate explicilty and check final residual
    if (dynamic) mat_vec.set_product_tol(0.1*tol/static_cast<double>(m));
    mat_vec(x, w[0]);
    // w[0] = b - w[0] = b - A*x
    w[0].EqualsAXPlusBY(1.0, b, -1.0, w[0]);

    double res = w[0].Norm2();
    fout << "# SteihaugFOM final (true) residual : |res|/|res0| = "
         << res/norm0 << endl;
    
    if (fabs(res - beta) > 0.01*tol*norm0) {
      fout << "# WARNING in SteihaugFOM: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = " << (res - beta)/norm0 << endl;
    }
  }
}
// ==============================================================================
template <class Vec>
void MINRES(int m, double tol, const Vec & b, Vec & x,
	    MatrixVectorProduct<Vec> & mat_vec,
	    Preconditioner<Vec> & precond, int & iters, 
            ostream & fout = cout, const bool & check_res = true,
            const bool & dynamic = false) {
  // check the subspace size
  if (m < 1) {
    cerr << "MINRES: illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }

  double Anorm = 0.0;
  double Acond = 0.0;
  double rnorm = 0.0;
  double ynorm = 0.0;
  x = 0.0;

  // Set up y and v for the first Lanczos vector v1.
  // y = beta0 P' v1, where P = C**(-1).
  // v is really P' v1.
  Vec y(b);
  Vec w(b);
  precond(w, y);
  Vec r1(b);
  double beta0 = InnerProd(b, y);
  iters = 1;

  if (beta0 < 0.0) {
    // preconditioner must be indefinite
    cerr << "MINRES: preconditioner provided is indefinite." << endl;
    throw(-1);
  }
  if (beta0 == 0.0) {
    // rhs must be zero, so stop with x = 0
    fout << "MINRES: rhs vector b = 0, so returning x = 0." << endl;
    return;
  }  
  beta0 = sqrt(beta0);
  rnorm = beta0;
  
  // should provide checks here to see if preconditioner and matrix are
  // symmetric
  if (dynamic) mat_vec.set_product_tol(tol/static_cast<double>(m));
  mat_vec(y, w);
  double Arnorm1 = w.Norm2();

  // initialized remaining quantities in preparation for iteration
  double oldb = 0.0;
  double beta = beta0;
  double dbar = 0.0;
  double oldeps = 0.0;
  double epsln = 0.0;
  double qrnorm = beta0;
  double phibar = beta0;
  double rhs1 = beta0;
  double rhs2 = 0.0;
  double tnorm2 = 0.0;
  double ynorm2 = 0.0;
  double cs = -1.0;
  double sn = 0.0;
  double gmax = 0.0;
  double gmin = 0.0;
  Vec w2(x); // recall x = 0
  Vec w1(x); // recall x = 0
  Vec r2(r1);
  Vec v;

  // output header information including initial residual
  int i = 0;
  writeKrylovHeader(fout, "MINRES", tol, beta0);
  writeKrylovHistory(fout, i, beta0, beta0);

  // loop over all serach directions
  for (i = 0; i < m; i++) {

    // Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
    // The general iteration is similar to the case k = 1 with v0 = 0:
    //
    //   p1      = Operator * v1  -  beta0 * v0,
    //   alpha1  = v1'p1,
    //   q2      = p2  -  alpha1 * v1,
    //   beta2^2 = q2'q2,
    //   v2      = (1/beta2) q2.
    //
    // Again, y = betak P vk,  where  P = C**(-1).
    v = y;
    v *= (1.0/beta);

    if (dynamic) mat_vec.set_product_tol(
            tol*beta0/(rnorm*static_cast<double>(m)));
    mat_vec(v, y);
    if (i > 0) y.EqualsAXPlusBY(1.0, y, -beta/oldb, r1); 
    
    double alpha = InnerProd(v, y);
    y.EqualsAXPlusBY(1.0, y, -alpha/beta, r2);
    r1 = r2;
    r2 = y;    
    precond(r2, y);

    oldb = beta;
    beta = InnerProd(r2, y);
    if (beta < 0.0) {
      cerr << "MINRES: matrix does not appear to be symmetric." << endl;
      throw(-1);
    }
    beta = sqrt(beta);
    tnorm2 = tnorm2 + alpha*alpha + oldb*oldb + beta*beta;

    if (i == 0) {
      // initialize a few things The following corresponds to y being the
      // solution of a generalized eigenvalue problem, Ay = lambda My.  This is
      // highly unlikely in our context, so we ignore this
      // if (beta/beta0 <= 10.0*kEpsilon) istop = -1; // beta2 = 0 or is ~ 0
      gmax = abs(alpha);
      gmin = gmax;
    }

    // Apply previous rotation Q_{k-1} to get
    //   [delta_k epsln_{k+1}] = [cs  sn][dbar_k    0       ]
    //   [gbar_k dbar_{k+1}]     [sn -cs][alpha_k beta_{k+1}].
    oldeps = epsln;
    double delta = cs*dbar + sn*alpha;
    double gbar  = sn*dbar - cs*alpha;
    epsln = sn*beta;
    dbar  = -cs*beta;

    // Compute the new plane rotation Q_k
    double gamma  = sqrt(gbar*gbar + beta*beta);
    cs     = gbar/gamma;
    sn     = beta/gamma;
    double phi    = cs*phibar;
    phibar = sn*phibar;
    
    // update solution
    double denom = 1.0/gamma;

    w1 = w2;
    w2 = w;
    w.EqualsAXPlusBY(denom, v, -oldeps*denom, w1);
    w.EqualsAXPlusBY(1.0, w, -delta*denom, w2);
    x.EqualsAXPlusBY(1.0, x, phi, w);
        
    gmax = std::max<double>(gmax, gamma);
    gmin = std::min<double>(gmin, gamma);
    double z = rhs1 /gamma;
    ynorm2 = z*z + ynorm2;
    rhs1 = rhs2 - delta*z;
    rhs2 = - epsln*z;

    // estimate various norms and test for convergence
    Anorm = sqrt(tnorm2);
    ynorm = sqrt(ynorm2);
    double epsx  = Anorm*ynorm*kEpsilon;

    qrnorm = phibar;
    rnorm = qrnorm;
    double rootl = sqrt(gbar*gbar + dbar*dbar);
    double relAnorm = rootl/Anorm;

    // Estimate  cond(A).
    // In this version we look at the diagonals of R in the factorization of the
    // lower Hessenberg matrix, Q * H = R, where H is the tridiagonal matrix
    // from Lanczos with one extra row, beta(k+1) e_k^T.
    Acond = gmax/gmin;

    // output relative residual norm and check stopping criteria
    writeKrylovHistory(fout, i+1, rnorm, beta0);
    iters++;
#if 0    
    if (Acond >= 0.1/kEpsilon) {
      fout << "MINRES stopping: estimate for cond(A) >= 0.1/eps." << endl;
      break;
    }
#endif
    if ( (rnorm <= epsx) || (relAnorm <= epsx) ) {
      fout << "MINRES stopping: residual has reached machine zero." << endl;
      break;
    }
    if ( rnorm <= beta0*tol ) break;
  }

  if (check_res) {
    // recalculate explicilty and check final residual
    mat_vec(x, w);
    // w = b - w = b - A*x
    w.EqualsAXPlusBY(1.0, b, -1.0, w);
    precond(w, y);
    iters++;
    double res = sqrt(InnerProd(w, y));
    fout << "# MINRES final (true) residual : |res|/|res0| = "
         << res/beta0 << endl;
    
    if (fabs(res - rnorm) > 0.01*tol*beta0) {
      fout << "# WARNING in MINRES: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - rnorm)/res0 = " << (res - rnorm)/beta0 << endl;
    }
  }
}
// ==============================================================================

template <class Vec>
void FITR_old(int m, double tol, double radius, const Vec & b, Vec & x,
              MatrixVectorProduct<Vec> & mat_vec,
              Preconditioner<Vec> & precond, int & iters, double & pred,
              bool & active, ostream & fout = cout, const bool & check = true,
              const bool & dynamic = false) {
  // check the subspace size
  if (m < 1) {
    cerr << "FITR: illegal value for subspace size, m = " << m << endl;
    throw(-1);
  }
  if (radius <= 0.0) {
    cerr << "FITR: trust-region radius must be positive, radius = "
         << radius << endl;
    throw(-1);
  }

  // define various vectors and arrays

  // Note: elements in v and z are initialized to x to avoid creating
  // a temporary Vec object from the copy constructor

  // Note: STL vectors are used for v and z while boost::numeric::ublas vectors
  // are used for double based vectors and matrices; this choice was made
  // mostly to help emphasize that the objects in v and z are fundamentally
  // different (they are some sort of vectors themselves).
  vector<Vec> v(m+1, x);
  vector<Vec> z(m, x);
  Vec r(x);
  ublas::vector<double> g(2*m+1, 0.0);
  ublas::vector<double> y(m, 0.0);
  ublas::matrix<double> B(m+1, m, 0.0);
  ublas::matrix<double> H(m+1, m, 0.0);
  iters = 0;

  // calculate the norm of the rhs vector and initialize the RHS of the reduced
  // system
  double norm0 = b.Norm2();
  v[0] = b;
  double beta = norm0;
#if 0
  if ( (beta < tol*norm0) || (beta < kEpsilon) ) {
    // system is already solved
    fout << "FITR system solved by initial guess." << endl;
    return;
  }
#endif
  g(0) = beta;
  
  // normalize residual to get v_{0} and output header information including
  // initial residual
  r = b;
  v[0] /= beta;
  int i = 0;
  //writeKrylovHistory(fout, i, beta, norm0);
  
  // begin flexible Arnoldi iterations
  bool lin_depend = false;
  bool reset = false;
  double lambda = 0.0;
  double lambda_1 = 0.0;
  double lambda_2 = 0.0;
  active = false;
  for (i = 0; i < m; i++) {
    iters++;

    // precondition the residual and store result in z[i]
#if 0
    precond.set_diagonal(std::max(
        lambda + (lambda - lambda_1) + (lambda - 2.0*lambda_1 + lambda_2), 0.0));
#endif
    precond.set_diagonal(std::max(lambda + 2.0*(lambda - lambda_1), 0.0));

    if (i == 0)
      z[i] = r;
    else
      precond(r, z[i]);

#if 0
    if (i == 0)
      z[i] = v[i];
    else
      precond(v[i], z[i]);
#endif
    //cout << "FITR: after precondition..." << endl;
    
    // Orthonormalize z[i] against previous z vectors
    if (i == 0) {
      // on first iteration, just need to normalize z[0]
      double nrm = z[0].Norm2();
      if (nrm <= 0.0) {
        cerr << "FITR: initial preconditioned vector is zero." << endl;
        throw(-1);
      }
      z[i] /= nrm;
    } else {
      try {
        modGramSchmidt(i-1, z);
      } catch (string err_msg) {
        cerr << "FITR: Z vector orthogonalization: " << err_msg << endl;
        throw(-1);
      } catch (bool depend) {
        lin_depend = depend;
      }
    }
    
    //cout << "FITR: after GS Z..." << endl;
    
    // add to Krylov subspace
    if (dynamic) mat_vec.set_product_tol(
            tol*norm0/(beta*static_cast<double>(m)));
    mat_vec(z[i], v[i+1]);
    
    // modified Gram-Schmidt orthogonalization
    try {
      modGramSchmidt(i, H, v);
    } catch (string err_msg) {
      cerr << "FITR: V vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      lin_depend = depend;
    }
    //cout << "FITR: after GS V..." << endl;
    
    // add new row and column to matrix B = V^{T} Z
    for (int k = 0; k <= i; k++) {
      B(k,i) = InnerProd(v[k], z[i]);
      B(i+1,k) = InnerProd(v[i+1], z[k]);
    }
    //cout << "FITR: after adding rows/columns to B..." << endl;
    
    // compute the (symmetric part of) reduced matrix F = B^T H and rhs = B^T g,
    // then solve the reduced Trust region problem
    ublas::matrix<double> F(i+1, i+1, 0.0);
    ublas::vector<double> rhs(i+1);
    for (int j = 0; j < i+1; j++) {
      for (int k = 0; k < i+1; k++) {
        //F(j,k) = 0.0;
        for (int l = 0; l < i+2; l++) {
          double tmp = 0.5*B(l,j)*H(l,k);
          F(j,k) += tmp;
          F(k,j) += tmp;
        }
      }
      rhs(j) = -g(0)*B(0,j); // -ve sign is because b = -g
    }
    lambda_2 = lambda_1;
    lambda_1 = lambda;
    solveTrustReduced(i+1, F, radius, rhs, y, lambda, pred);
    if (lambda > kEpsilon) active = true;

#if 0
    // compute the residual norm
    ublas::matrix_range<ublas::matrix<double> >
        H_r(H, ublas::range(0,i+2), ublas::range(0,i+1)),
        B_r(B, ublas::range(0,i+2), ublas::range(0,i+1));
    beta = trustResidual(i+1, H_r, B_r, g, y, lambda);
#endif

    // compute the residual norm
    r = b;
    for (int k = 0; k < i+1; k++) {
      // r = r - lambda*y[k]*z[k]
      r.EqualsAXPlusBY(1.0, r, -lambda*y[k], z[k]);
    }
    ublas::vector<double> Hy(i+2,0.0);
    for (int k = 0; k < i+2; k++) {
      Hy(k) = 0.0;
      for (int j = 0; j < i+1; j++)
        Hy(k) += H(k,j)*y(j);
    }
    for (int k = 0; k < i+2; k++) {
      // r = r - V * H * y
      r.EqualsAXPlusBY(1.0, r, -Hy(k), v[k]);
    }    
    beta = r.Norm2();

#if 0
    if ( (lambda > kEpsilon) && (!reset) ) {
      norm0 = beta;
      reset = true;
      cout << "new norm0 = " << norm0 << endl;
    }
#endif
    if (i == 0) {
      boost::format col_head("%|5| %|8t| %|-12| %|20t| %|-12|\n");
      col_head % "# iter" % "rel. res." % "lambda";
      writeKrylovHeader(fout, "FITR", tol, beta, col_head);
      norm0 = beta;
    }
    fout << boost::format("%|5| %|8t| %|-12.6| %|20t| %|-12.6|\n")
        % (i+1) % (beta/norm0) % lambda;
    
    // check for convergence
    if ( (beta < tol*norm0) || (beta < kEpsilon) ) { //1.e-12) ) {
      break;
    } else if (lin_depend) {
      cerr << "FITR: one of z[" << i << "] or  v[" << i+1 << "]"
           << " is linearly dependent, and residual is nonzero." << endl;
      cerr << "      ||res|| = " << beta << endl;
      throw(-1);
    }
  }

  // update the solution
  x = 0.0;
  for (int k = 0; k < std::min(i+1,m); k++) {
    // x = x + y[k]*z[k]
    x.EqualsAXPlusBY(1.0, x, y[k], z[k]);
  }

  if (check) {
    // check the residual norm
    if (dynamic) mat_vec.set_product_tol(0.1*tol/static_cast<double>(m));
    mat_vec(x, v[0]);
    v[0].EqualsAXPlusBY(1.0, v[0], lambda, x);    
    // v[0] = b - v[0] = b - (lambdaI + A)*x
    v[0].EqualsAXPlusBY(1.0, b, -1.0, v[0]);
    double res = v[0].Norm2();
    fout << "# FITR final (true) residual : |res|/|res0| = "
         << res/norm0 << endl;    
    if (fabs(res - beta) > 0.01*tol*norm0) {
      fout << "# WARNING in FITR: true residual norm and calculated "
           << "residual norm do not agree." << endl;
      fout << "# (res - beta)/res0 = " << (res - beta)/norm0 << endl;
    }
    // check that solution satisfies the trust region
    double x_norm = x.Norm2();
    if (fabs(lambda) < kEpsilon) {
      if (x_norm > radius) {
        cerr << "FITR: lambda = 0 and solution is not inside trust region."
             << endl;
        throw(-1);
      }
    } else { // solution should lie on trust region
      if (x_norm - radius > 1.e-6*radius) {
        cerr << "FITR: lambda > 0 and solution is not on trust region." << endl;
        cerr << "x_norm - radius = " << x_norm - radius << endl;     
        throw(-1);            
      }    
    }
  }
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
TRISQP<Vec, PrimVec, DualVec>::TRISQP() {
  maxiter_ = -1;
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void TRISQP<Vec, PrimVec, DualVec>::SubspaceSize(int m) {
  // check the subspace size
  if (m < 1) {
    cerr << "TRISQP::SubspaceSize(): illegal value for subspace size, m = " << m
         << endl;
    throw(-1);
  }
  maxiter_ = m;
  
  // Note: STL vectors are used for V, VTilde, Lambda, and LambdaTilde while
  // boost::numeric::ublas vectors are used for double based vectors and
  // matrices; this choice was made mostly to help emphasize that the objects in
  // V, VTilde, etc are fundamentally different (they are some sort of vectors
  // themselves).

  V_.clear();
  V_.reserve(maxiter_+1);
  VTilde_.clear();
  VTilde_.reserve(maxiter_);
  Z_.clear();
  Z_.reserve(maxiter_+1);
  Lambda_.clear();
  Lambda_.reserve(maxiter_+1);
  
  f_.resize(maxiter_+1, 0.0);
  g_.resize(maxiter_+1, 0.0);
  y_.resize(maxiter_, 0.0);
  sig_.resize(maxiter_, 0.0);
  B_.resize(maxiter_+1, maxiter_, 0.0);
  C_.resize(maxiter_+1, maxiter_, 0.0);
  D_.resize(maxiter_+1, maxiter_+1, 0.0);
  Vprods_.resize(maxiter_, maxiter_+1, 0.0);
  VZprods_.resize(maxiter_, maxiter_, 0.0);
}
// ==============================================================================
#if 0
template <class Vec, class PrimVec, class DualVec>
void TRISQP<Vec, PrimVec, DualVec>::BuildBasis(
    double tol, const Vec & b, Vec & x, MatrixVectorProduct<Vec> & mat_vec,
    Preconditioner<Vec> & precond, int & iters, ostream & fout) {
  if (maxiter_ < 1) {
    cerr << "TRISQP::BuildBasis(): SubspaceSize must be called first." << endl;
    throw(-1);
  }
  if (tol < 0.0) {
    cerr << "TRISQP::BuildBasis(): tolerance must be positive, tol = "
         << tol << endl;
    throw(-1);
  }

  // set initial dual vector (Lambda_[0] = c/gamma)
  Lambda_[0] = 1.0; //b.dual();
  double feas_norm0 = b.dual().Norm2();
  double gamma = Lambda_[0].Norm2();
  Lambda_[0] /= gamma;

#if 0
  // set initial primal vector (V_[0] = g/beta)
  V_[0] = 1.0; //b.primal();
  double grad_norm0 = V_[0].Norm2(); //b.primal().Norm2();
  double beta = grad_norm0;
  V_[0] /= beta;
#endif
  
  // clear vectors and matrices
  y_.clear();
  f_.clear();
  sig_.clear();
  B_.clear();
  C_.clear();
  
  bool V_lin_dep = false;
  bool Lambda_lin_dep = false;
  int j;
  iters = 0;
  for (j = 0; j < maxiter_; j++) {
    iters++;


    // expand primal subspace
    r_.dual() = Lambda_[j];
    r_.primal() = 0.0;
    mat_vec(r_, x);
    V_[j] = x.primal();
    try {      
      //modGramSchmidt(j-1, B_, V_);
      // TEMP: begin main Gram-Schmidt loop
      for (int k = 0; k < j; k++) {
        double prod = InnerProd(V_[k], V_[j]);
        B_(k,j) = prod;
        V_[j].EqualsAXPlusBY(1.0, V_[j], -prod, V_[k]);
      }
      B_(j,j) = V_[j].Norm2();
      V_[j] /= B_(j,j);
    } catch (string err_msg) {
      cerr << "TRISQP: V vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP: V linear dependence" << endl;
      V_lin_dep = depend;
    }
    
    // expand dual subspace
    r_.primal() = V_[j];
    r_.dual() = 0.0;
    mat_vec(r_, x);
    Lambda_[j+1] = x.dual();
    try {
      modGramSchmidt(j, C_, Lambda_);
    } catch (string err_msg) {
      cerr << "TRISQP: Lambda vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP: Lambda linear dependence" << endl;
      Lambda_lin_dep = depend;
    }

    // solve the normal subproblem
    ublas::matrix<double> A = ublas::prod(
        ublas::project(C_, ublas::range(0,j+2), ublas::range(0,j+1)),
        ublas::project(B_, ublas::range(0,j+1), ublas::range(0,j+1)));

#if 0
    cout << "C_ = " << endl;
    for (int k = 0; k < j+2; k++){
      for (int l = 0; l < j+1; l++)
        cout << C_(k,l) << " ";
      cout << endl;
    }
    cout << "B_ = " << endl;
    for (int k = 0; k < j+1; k++) {
      for (int l = 0; l < j+1; l++)
        cout << B_(k,l) << " ";
      cout << endl;
    }
    cout << "A = " << endl;
    for (int k = 0; k < j+2; k++) {
      for (int l = 0; l < j+1; l++)
        cout << A(k,l) << " ";
      cout << endl;
    }
#endif
    
    ublas::vector<double> rhs(j+2, 0.0), z(j+1, 0.0);
    rhs(0) = -feas_norm0;
    f_(j) = -InnerProd(Lambda_[j], b.dual()); 
#if 0
    ublas::vector<double> Sigma, P, QT;
    computeSVD(j+2, j+1, A, Sigma, P, QT);
    ublas::matrix<double> Pmat(j+2, j+1, 0.0),
        QTmat(j+1, j+1, 0.0);
    for (int l = 0; l < j+1; l++) {
      for (int k = 0; k < j+2; k++)
        Pmat(k,l) = P(l*(j+2)+k);
      for (int k = 0; k < j+1; k++)
        QTmat(k,l) = QT(l*(j+1)+k);
    }    
    solveLeastSquaresOverSphere(j+2, j+1, 1e32, Sigma, Pmat, QTmat, rhs,
                                z);
    ublas::project(sig_, ublas::range(0,j+1)) = z;
    ublas::project(y_, ublas::range(0,j+1)) = -ublas::prod(
        ublas::project(B_, ublas::range(0,j+1), ublas::range(0,j+1)), z);

    // compute the residual-norm of the normal subproblem
    ublas::vector<double> res_red = ublas::project(f_, ublas::range(0,j+2))
        - ublas::prod(A, z);
    gamma = norm_2(res_red);
    cout << "iter = " << j << " relative res = " << gamma/feas_norm0
         << endl;
#endif
    
    ublas::vector<double> Sigma, P, QT;
    computeSVD(j+1, j+1, A, Sigma, P, QT);
    ublas::matrix<double> Pmat(j+1, j+1, 0.0),
        QTmat(j+1, j+1, 0.0);
    for (int l = 0; l < j+1; l++) {
      for (int k = 0; k < j+1; k++) {
        Pmat(k,l) = P(l*(j+1)+k); 
        QTmat(k,l) = QT(l*(j+1)+k);
      }
    }    
    solveLeastSquaresOverSphere(j+1, j+1, 1e32, Sigma, Pmat, QTmat, f_,
                                z);
    ublas::project(sig_, ublas::range(0,j+1)) = z;
    ublas::project(y_, ublas::range(0,j+1)) = -ublas::prod(
        ublas::project(B_, ublas::range(0,j+1), ublas::range(0,j+1)), z);

    // compute the residual-norm of the normal subproblem
    r_.dual() = b.dual();;
    ublas::vector<double> Az = ublas::prod(A, z);
    for (int k = 0; k < j+2; k++)
      r_.dual().EqualsAXPlusBY(1.0, r_.dual(), Az(k), Lambda_[k]);
    gamma = r_.dual().Norm2();;
    cout << "iter = " << j << " relative res = " << gamma/feas_norm0
         << endl;

    if (gamma < feas_norm0*tol) break;


#if 0
    // expand subspace
    r_.primal() = V_[j];
    r_.dual() = Lambda_[j];
    mat_vec(r_, x);
    V_[j+1] = x.primal();
    Lambda_[j+1] = x.dual();

    // Orthogonalize V_[j+1] and Lambda_[j+1] against previous vectors, and
    // compute the matrices B_ and C_
    try {
      modGramSchmidt(j, C_, Lambda_);
    } catch (string err_msg) {
      cerr << "TRISQP: Lambda vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP: Lambda linear dependence" << endl;
      Lambda_lin_dep = depend;
    }
    try {
      modGramSchmidt(j, B_, V_);
    } catch (string err_msg) {
      cerr << "TRISQP: V vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP: V linear dependence" << endl;
      V_lin_dep = depend;
    }
    
    // solve the normal subproblem
    ublas::matrix<double> A(2*(j+2),2*(j+1), 0.0);
    ublas::project(A, ublas::range(0,j+2), ublas::range(j+1,2*(j+1)))
        = ublas::project(B_, ublas::range(0,j+2), ublas::range(0,j+1));
    ublas::project(A, ublas::range(j+2,2*(j+2)), ublas::range(0,j+1))
        = ublas::project(C_, ublas::range(0,j+2), ublas::range(0,j+1));
    for (int k = 0; k <= j; k++)
      A(k,k) = 1.0;

#if 0
    cout << "A = " << endl;
    for (int k = 0; k < 2*(j+2); k++) {
      for (int l = 0; l < 2*(j+1); l++)
        cout << A(k,l) << " ";
      cout << endl;
    }
#endif

    ublas::vector<double> rhs(2*(j+2), 0.0), z(2*(j+1), 0.0);
    rhs(j+2) = feas_norm0;

    ublas::vector<double> Sigma, P, QT;
    computeSVD(2*(j+2), 2*(j+1), A, Sigma, P, QT);
    ublas::matrix<double> Pmat(2*(j+2), 2*(j+1), 0.0),
        QTmat(2*(j+1), 2*(j+1), 0.0);
    for (int l = 0; l < 2*(j+1); l++) {
      for (int k = 0; k < 2*(j+2); k++)
        Pmat(k,l) = P(l*2*(j+2)+k);
      for (int k = 0; k < 2*(j+1); k++)
        QTmat(k,l) = QT(l*2*(j+1)+k);
    }    
    solveLeastSquaresOverSphere(2*(j+2), 2*(j+1), 1e32, Sigma, Pmat, QTmat, rhs,
                                z);
    ublas::project(y_, ublas::range(0,j+1)) =
        ublas::project(z, ublas::range(0,j+1));
    ublas::project(sig_, ublas::range(0,j+1)) =
        ublas::project(z, ublas::range(j+1,2*(j+1)));

    // compute the residual-norm of the normal subproblem
    ublas::vector<double> res_red = rhs - ublas::prod(A, z);
    gamma = norm_2(res_red);
    cout << "iter = " << j << " relative res = " << gamma/feas_norm0
         << endl;

    if (gamma < feas_norm0*tol) break;
#endif
    
  }

  // build the solution to the normal problem
  x = 0.0;
  for (int k = 0; k < iters; k++) {
    x.primal().EqualsAXPlusBY(1.0, x.primal(), y_(k), V_[k]);
    x.dual().EqualsAXPlusBY(1.0, x.dual(), sig_(k), Lambda_[k]);
  }

  bool check = true;
  if (check) {
    mat_vec(x, r_);
    r_.primal() += x.primal();
    r_.dual() -= b.dual();
    cout << "TRISQP: computed residual norm = " << gamma/feas_norm0 << endl;
    cout << "TRISQP: actual residual norm   = "
         << r_.dual().Norm2()/feas_norm0 << endl;
#if 0
    // how much does a vector in the "null space" impace the feasibility
    double sol_norm = x.primal().Norm2();
    cout << "solution norm = " << sol_norm << endl;
    x.primal().EqualsAXPlusBY(1.0, x.primal(), sol_norm/b.primal().Norm2(),
                              b.primal());
    mat_vec(x, r_);
    r_.primal() += x.primal();
    r_.dual() -= b.dual();
    cout << "TRISQP: residual norm after pertubation = "
         << r_.dual().Norm2()/feas_norm0 << endl;
#endif    
    // how well does V_ represent the range of A?
    x = b; //1.0;
    x.dual() = 0.0;
    mat_vec(x, r_);
    double init = r_.dual().Norm2();
    
    for (int k = 0; k < iters; k++) {
      double tmp = InnerProd(x.primal(), V_[k]);
      x.primal().EqualsAXPlusBY(1.0, x.primal(), -tmp, V_[k]);
    }
    mat_vec(x, r_);
    cout << "TRISQP: norm of solution = " << x.primal().Norm2() << endl;
    cout << "TRISQP: quality of projection = " << r_.dual().Norm2()/init << endl;

    
    
    // how well does V_ represent the range of A?
    x = 1.0;
    x.dual() = 0.0;
    mat_vec(x, r_);
    init = r_.dual().Norm2();
    
    for (int k = 0; k < iters; k++) {
      double tmp = InnerProd(x.primal(), V_[k]);
      x.primal().EqualsAXPlusBY(1.0, x.primal(), -tmp, V_[k]);
    }
    mat_vec(x, r_);
    cout << "TRISQP: norm of solution = " << x.primal().Norm2() << endl;
    cout << "TRISQP: quality of projection = " << r_.dual().Norm2()/init << endl;

#if 0
    // another check on how well V_ represents range of A
    x = 1.0;
    x.dual() = 0.0;
    ublas::vector<double> VTz(iters+1, 0.0);
    for (int k = 0; k < iters+1; k++)
      VTz(k) = InnerProd(x.primal(), V_[k]);
    ublas::matrix<double> rhs(iters, 1, 0.0);
    for (int k = 0; k < iters; k++) {
      for (int j = 0; j < iters+1; j++)
        rhs(k,1) += B_(j,k)*VTz(j);
    }
    ublas::matrix<double> A = ublas::project(C_, ublas::range(0,iters),
                                             ublas::range(0,iters));
    ublas::matrix<double> y(iters, 1, 0.0);
    solveReducedMultipleRHS(iters, A, 1, rhs, y);
    for (int k = 0; k < iters; k++)
      x.primal().EqualsAXPlusBY(1.0, x.primal(), -y(k,1), V_[k]);
    mat_vec(x, r_);
    cout << "TRISQP: norm of solution = " << x.primal().Norm2() << endl;
    cout << "TRISQP: quality of projection = " << r_.dual().Norm2()/init << endl;
#endif
    
#if 0
    // Find eigenvectors associated with zero eigenvalues of A^T*A
    ublas::matrix<double> A(iters-1,iters-1,0.0);
    for (int j = 0; j < iters-1; j++)      
      for (int k = 0; k < iters-1; k++)
        for (int l = 0; l < iters; l++)
          A(j,k) += B_(j,l)*C_(l,k);
    ublas::vector<double> eig(iters-1, 0.0);
    eigenvalues(iters-1, A, eig);
    cout << "eig = ";
    for (int j = 0; j < iters-1; j++)
      cout << eig(j) << " ";
    cout << endl;
    for (int j = 0; j < iters-1; j++)
      A(j,j) -= eig(iters-2);
    ublas::matrix<double> EVec(iters-1,iters-1, 0.0);
    eigenvaluesAndVectors(iters-1, A, eig, EVec);
    cout << "eig = ";
    for (int j = 0; j < iters-1; j++)
      cout << eig(j) << " ";
    cout << endl;

    x = 0.0;
    for (int k = 0; k < iters-1; k++)
      x.primal().EqualsAXPlusBY(1.0,x.primal(),EVec(k,0),V_[k]);
    mat_vec(x, r_);
    cout << "TRISQP: norm of eigenvector   = " << x.primal().Norm2() << endl;
    cout << "TRISQP: quality of projection = " << r_.dual().Norm2() << endl; 
#endif
    
  }

#if 0
  // Try using Arnoldi (Lanczos) to find null space
  B_.clear();
  V_lin_dep = false;
  Lambda_lin_dep = false;
  iters = 0;
  for (j = 0; j < maxiter_; j++) {
    iters++;

    // expand Krylov subspace
    r_.dual() = 0.0;
    r_.primal() = V_[j];
    mat_vec(r_, x);
    mat_vec(x, r_);
    V_[j+1].EqualsAXPlusBY(1.0, r_.primal(), -10.0, V_[j]);

    try {
      modGramSchmidt(j, B_, V_);
    } catch (string err_msg) {
      cerr << "TRISQP: V vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP: V linear dependence" << endl;
      V_lin_dep = depend;
    }
    
    ublas::matrix<double> A = ublas::project(B_, ublas::range(0,j+1),
                                             ublas::range(0,j+1));
    ublas::vector<double> eig(j+1, 0.0);
    ublas::matrix<double> EVec(j+1,j+1, 0.0);
    eigenvaluesAndVectors(j+1, A, eig, EVec);
    cout << "eig = ";
    for (int k = 0; k < j+1; k++)
      cout << eig(k) << " ";
    cout << endl;

    x = 0.0;
    for (int k = 0; k < j+1; k++)
      x.primal().EqualsAXPlusBY(1.0, x.primal(), EVec(k,0), V_[k]);
    mat_vec(x, r_);
    cout << "TRISQP: norm of eigenvector 1   = " << x.primal().Norm2() << endl;
    cout << "TRISQP: quality of projection 1 = " << r_.dual().Norm2() << endl;
    x = 0.0;
    if (j > 0) {
      for (int k = 0; k < j+1; k++)
        x.primal().EqualsAXPlusBY(1.0, x.primal(), EVec(k,1), V_[k]);
      mat_vec(x, r_);
      cout << "TRISQP: norm of eigenvector 2   = " << x.primal().Norm2() << endl;
      cout << "TRISQP: quality of projection 2 = " << r_.dual().Norm2() << endl;
    }
    if (j > 1) {
      x = 0.0;
      for (int k = 0; k < j+1; k++)
        x.primal().EqualsAXPlusBY(1.0, x.primal(), EVec(k,2), V_[k]);
      mat_vec(x, r_);
      cout << "TRISQP: norm of eigenvector 3   = " << x.primal().Norm2() << endl;
      cout << "TRISQP: quality of projection 3 = " << r_.dual().Norm2() << endl;
    }
  }
#endif
  
#if 0
  W_[0].dual() = b.dual();
  W_[0].primal() = 0.0;
  // calculate the norm of the rhs vector and initialize the RHS of the reduced
  // system
  double feas_norm0 = b.dual().Norm2();
  double gamma = feas_norm0;  
  ublas::matrix<double> rhs(maxiter_, maxiter_, 0.0);
  rhs(0,0) = gamma;
  W_[0] /= gamma;

  bool V_lin_dep = false;
  bool Lambda_lin_dep = false;
  int j;
  iters = 0;
  for (j = 0; j < maxiter_; j++) {
    iters++;

    // precondition
    r_.primal() = W_[j].primal(); 
    r_.dual() = W_[j].dual();
    //precond(r_, x);
    x = r_;
    VTilde_[j] = x.primal();
    LambdaTilde_[j] = x.dual();

#if 0
    // Orthonormalize VTilde_[i] and LambdaTilde_[i] against previous vectors
    try {
      modGramSchmidt(j-1, VTilde_);
    } catch (string err_msg) {
      cerr << "TRISQP::BuildBasis(): "
           << "VTilde vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      V_lin_dep = depend;
    }
    try { // orthogonalize the LambdaTilde_[i]
      modGramSchmidt(j-1, LambdaTilde_);
    } catch (string err_msg) {
      cout << "TRISQP::BuildBasis(): "
           << "LambdaTilde vector orthogonalization: " << err_msg << endl;
      //throw(-1);
    } catch (bool depend) {
      Lambda_lin_dep = depend;
    }
#endif
    
    // expand subspace
    r_.primal() = VTilde_[j];
    r_.dual() = LambdaTilde_[j];
    mat_vec(r_, W_[j+1]);

    // Orthogonalize W_[j+1] against previous vectors, and compute matrix D_
    try {
      modGramSchmidt(j, D_, W_);
    } catch (string err_msg) {
      cerr << "TRISQP::BuildBasis(): "
           << "Lambda vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP::BuildBasis(): "
           << "Lambda linear dependence" << endl;
      Lambda_lin_dep = depend;
    }

    // Add to RHS of reduced problem
    if (j > 0) {
      for (int k = 0; k <= j; k++) {
        rhs(k,j) = InnerProd(W_[k].primal(), W_[j].primal());
        if (k > 0) rhs(j,k) = rhs(k,j);
      }
    }

    cout << "rhs = " << endl;
    for (int k = 0; k <= j; k++) {
      for (int l = 0; l <= j; l++)
        cout << rhs(k,l) << " ";
      cout << endl;
    }
    
#if 0
    cout << "D_ = " << endl;
    for (int k = 0; k <= j; k++) {
      for (int l = 0; l <= j; l++)
        cout << D_(k,l) << " ";
      cout << endl;
    }
#endif
    
    // Solve reduced problem for Y
    solveReducedMultipleRHS(j+1, D_, j+1, rhs, Y_);
    
    // compute the residuals
    ublas::vector<double> res(j+1, 0.0);
    ublas::matrix_range<ublas::matrix<double> >
        D_r(D_, ublas::range(0,j+2), ublas::range(0,j+1)),
        Y_r(Y_, ublas::range(0,j+1), ublas::range(0,j+1));    
    ublas::matrix<double> DY = ublas::prod(D_r, Y_r);
    for (int k = 0; k <= j; k++) {
      r_ = 0.0;
      for (int l = 0; l <= j+1; l++)
        r_.EqualsAXPlusBY(1.0, r_, DY(l,k), W_[l]);
      if (k == 0) {
        r_.EqualsAXPlusBY(1.0, r_, -gamma, W_[0]);
        res(k) = r_.Norm2();
      } else {
        r_.primal().EqualsAXPlusBY(1.0, r_.primal(), -1.0, W_[k].primal());
        res(k) = r_.dual().Norm2();
      }
    }
    double max_res = 0.0;
    cout << "norm residuals = ";
    for (int k = 1; k <= j; k++) {
      max_res = std::max(max_res, res(k));
      cout << res(k) << " ";
    }
    cout << endl;
    cout << "iter = " << j << ": normal res = " << res(0)/feas_norm0
         << ": max norm res = " << max_res << endl;
    
#if 0
    // Orthonormalize the reduced-space solution
    ublas::matrix<double> gs(j+1, j+1, 0.0);
    for (int k = 0; k <= j; k++) {
      for (int l = 0; l < k; l++) {
        gs(l,k) = 0.0;
        for (int i = 0; i < j; i++)
          gs(l,k) += Y(i,k)*Y(i,l);
        for (int i = 0; i < j; i++)
          Y(i,k) -= gs(l,k)*Y(i,l);
      }
      gs(k,k) = 0.0;
      for (int i = 0; i < j; i++)
        gs(k,k) += Y(i,k)*Y(i,k);
      gs(k,k) = sqrt(gs(k,k));
      for (int i = 0; i < j; i++)
        Y(i,k) /= gs(k,k);
    }
    
#if 1
    // Check on Y(i,k)
    for (int k = 0; k <= j; k++) {
      for (int l = 0; l <= j; l++) {
        double tmp = 0.0;
        for (int i = 0; i < j; i++)
          tmp += Y(i,k)*Y(i,l);
        if ( ( (l == k) && (fabs(tmp - 1.0) > 1e-12) ) ||
             ( (l != k) && (fabs(tmp) > 1e-12) ) ) {
          cerr << "TRISQP::BuildBasis(): "
               << "Problem with Y orthonormalization" << endl;
          throw(-1);
        }
      }
    }
#endif

#endif
              
  }
    
  bool check = true;
  if (check) {
    for (int k = 1; k < iters; k++) {
      x = 0.0;
      for (int l = 0; l < iters; l++)
        x.primal().EqualsAXPlusBY(1.0, x.primal(), Y_(l,k), VTilde_[l]);      
      mat_vec(x, r_);
      cout << "null-space basis vector " << k << " norm = "
           << x.primal().Norm2() << ": error = " << r_.dual().Norm2() << endl;
      cout << "x.Norm2() = " << x.Norm2() << endl;
    }
    
  }
#endif
}
#endif
// ==============================================================================
//class ReducedKKTProduct;
#if 0
template <class Vec, class PrimVec, class DualVec>
void TRISQP<Vec, PrimVec, DualVec>::Solve(
    double grad_tol, double feas_tol, double radius, const Vec & b, Vec & x,
    MatrixVectorProduct<Vec> & mat_vec, Preconditioner<Vec> & precond,
    int & iters, double & mu, double & pred, bool & active, ostream & fout,
    const bool & check, const bool & dynamic) {
  if (maxiter_ < 1) {
    cerr << "TRISQP::Solve(): SubspaceSize must be called first." << endl;
    throw(-1);
  }
  if ( (grad_tol < 0.0) || (feas_tol < 0.0) ) {
    cerr << "TRISQP::Solve(): tolerances must be positive, grad_tol = "
         << grad_tol << ": feas_tol = " << feas_tol << endl;
    throw(-1);
  }
  if (radius <= 0.0) {
    cerr << "TRISQP::Solve(): trust-region radius must be positive, radius = "
         << radius << endl;
    throw(-1);
  }
  
  // The variable names below try to follow the paper as closely as possible,
  // although some discrepancies are inevitable
  r_ = b;
  V_[0] = b.primal();
  //Lambda_[0] = b.dual();
  iters = 0;
  
  // calculate the norm of the rhs vector and initialize the RHS of the reduced
  // system
  double grad_norm0 = b.primal().Norm2();
  double feas_norm0 = b.dual().Norm2();
  double beta = grad_norm0;
  double gamma = feas_norm0;
  V_[0] /= beta; // note, V_[0] is set to b.primal() when initialized
  g_.clear();
  g_(0) = beta;
  
  //Lambda_[0] /= gamma; // note, Lambda_[0] is set to b.dual() when initialized
  f_.clear();
  if (feas_tol >= 1.0) {
    // the constraints are satisfied, so assume that b.dual() = 0.0 and we need
    // to find an alternative to Lambda_[0]
    Lambda_[0] = 1.0;
    double tmp = Lambda_[0].Norm2();
    Lambda_[0] /= tmp;
  } else {
    Lambda_[0] = b.dual();
    Lambda_[0] /= gamma;
    f_(0) = gamma;    
  }

  double norm_radius = radius/sqrt(2.0); // trust-region radius for normal step

  // initialize reduced solution
  y_.clear();
  yt_.clear();
  yn_.clear();
  sig_.clear();
  B_.clear();
  C_.clear();
  D_.clear();
  Vprods_.clear();
  Lambdaprods_.clear();
  ublas::vector<double> y_tang;
  
  // nu is the Lagrange multiplier for the tangential-step subproblem
  double nu = 0.0;
  double old_nu = 0.0;
  bool V_lin_dep = false;
  bool Lambda_lin_dep = false;
  int j = 0;
  active = false;
  for (j = 0; j < maxiter_; j++) {
    iters++;

    // precondition the residual and store result in z[i]
    precond.set_diagonal(std::max(nu + (nu-old_nu),0.0));
    if (j == 0) {
      VTilde_[j] = V_[j];
      LambdaTilde_[j] = Lambda_[j];
    } else {
      // use x to store result of preconditioning; restarting not possible
      //precond(r, x); // *** using the residual seems to be a problem (too small?)
      
      // IDEA: set VTilde_[j] to a vector that is as close as possible to the null-space
      r_.primal() = V_[j];  // TEMP
      r_.dual() = Lambda_[j]; // TEMP
      //r_.primal()       
      precond(r_, x);
      VTilde_[j] = x.primal();
      LambdaTilde_[j] = x.dual();
#if 0
      // orthogonalize VTilde_[j] against Z_[1:j-1]
      for (int k = 0; k < j; k++) {
        if (D_(k,k) < 1.e-12) continue;
        double prod = InnerProd(VTilde_[j], Z_[k]);
        VTilde_[j].EqualsAXPlusBY(1.0, VTilde_[j], -prod, Z_[k]);
      }
#endif
      cout << "VTilde_[" << j << "].Norm2() = " << VTilde_[j].Norm2() << endl;
    }
    
    //cout << "TRISQP: after preconditioning" << endl;
    
    // Orthonormalize VTilde_[i] and LambdaTilde_[i] against previous vectors
    try {
      modGramSchmidt(j-1, VTilde_);
    } catch (string err_msg) {
      cerr << "TRISQP: VTilde vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      V_lin_dep = depend;
    }
    try { // orthogonalize the LambdaTilde_[i]
      modGramSchmidt(j-1, LambdaTilde_);
    } catch (string err_msg) {
      cout << "TRISQP: LambdaTilde vector orthogonalization: " << err_msg
           << endl;
      //throw(-1);
    } catch (bool depend) {
      Lambda_lin_dep = depend;
    }

    //cout << "TRISQP: after Gram-Schmidt on VTilde and LambdaTilde..." << endl;
    
    // Expand the subspace
    //if (dynamic) mat_vec.set_product_tol(
    //        tol*norm0/(beta*static_cast<double>(m)));
    r_.primal() = VTilde_[j];
    r_.dual() = LambdaTilde_[j];
    mat_vec(r_, x);
    
    // TEMP: try adding I to H
    //x.primal() += VTilde_[j];
    
    V_[j+1] = x.primal();
    Lambda_[j+1] = x.dual();
    cout << "V_[j+1].Norm2() (before normalization) = " << V_[j+1].Norm2() << endl;
    
    //cout << "TRISQP: after matrix-vector product..." << endl;
    
    // Orthogonalize V_[j+1] and Lambda_[j+1] against previous vectors, and
    // compute the matrices B_ and C_
    try {
      modGramSchmidt(j, C_, Lambda_);
    } catch (string err_msg) {
      cerr << "TRISQP: Lambda vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP: Lambda linear dependence" << endl;
      Lambda_lin_dep = depend;
    }
    // if V_[j+1] is small, and Lambda_[j+1] is linearly dependent, it is
    // likekly that the Hessian of the Lagrangian has zero eigenvalues; break
    if (Lambda_lin_dep)
      if (V_[j+1].Norm2() < kEpsilon) break;    
    try {
      modGramSchmidt(j, B_, V_);
    } catch (string err_msg) {
      cerr << "TRISQP: V vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP: V linear dependence" << endl;
      V_lin_dep = depend;
    }
    
    //cout << "TRISQP: after Gram-Schmidt on V and Lambda..." << endl;
    
    // add new row and column to matrices V_prods = VTilde^T V and
    // Lambda_prods = LambdaTilde^T Lambda
    for (int k = 0; k <= j; k++) {
      Vprods_(j,k) = InnerProd(VTilde_[j], V_[k]);
      Vprods_(k,j+1) = InnerProd(VTilde_[k], V_[j+1]);
      Lambdaprods_(j,k) = InnerProd(LambdaTilde_[j], Lambda_[k]);
      Lambdaprods_(k,j+1) = InnerProd(LambdaTilde_[k], Lambda_[j+1]);
    }

#if 0
    // expand the Z subspace
    r_.primal() = 0.0;
    r_.dual() = LambdaTilde_[j];
    mat_vec(r_, x);
    Z_[j] = x.primal();
    // TEMP: need to generalize Gram-Schmidt
#if 0
    for (int k = 0; k <= j; k++) { // bad idea; introduces terms not in Range(A^T)
      double prod = InnerProd(Z_[j], VTilde_[k]);
      Z_[j].EqualsAXPlusBY(1.0, Z_[j], -prod, VTilde_[k]);      
    }
#endif
    for (int k = 0; k < j; k++) {
      D_(k,j) = InnerProd(Z_[j], Z_[k]);
      Z_[j].EqualsAXPlusBY(1.0, Z_[j], -D_(k,j), Z_[k]);
    }
    D_(j,j) = Z_[j].Norm2();
    cout << "Norm of Z_[j] = " << D_(j,j) << endl;
    if (D_(j,j) > 1e-12) Z_[j] /= D_(j,j);
#endif
    
    // compute the SVD of C to find the null space approximation
    ublas::vector<double> Sigma, P, QT;
    computeSVD(j+2, j+1, C_, Sigma, P, QT);

#if 0
    // determine the dimensions of the normal and tangential subspaces
    norm_dim_ = 0;
    for (int k = 0; k < (int)(j+2)/2; k++) {
      if (Sigma(k) < sqrt(kEpsilon)*Sigma(0)) break;  // what if no constraints?
      norm_dim_++;
    }
    tang_dim_ = (j+1) - norm_dim_;
#elif 1
    // determine the dimensions of the normal and tangential subspaces
    norm_dim_ = 0;
    for (int k = 0; k < j+1; k++) {
      if (Sigma(k) < 0.01*Sigma(0)) break; //std::max(0.5*Sigma(0),0.5)) break;  // what if no constraints?
      norm_dim_++;
    }
    tang_dim_ = (j+1) - norm_dim_;
#else
    if (j == 0) {
      norm_dim_ = 1;
      tang_dim_ = 0;
    }
    else {
      if (beta*feas_norm0 > gamma*grad_norm0)
        tang_dim_++;
      else
        norm_dim_++;
    }
#endif

    
    // move QT and P into ublas::matrix format (to facilitate readability)
    ublas::matrix<double> QT_Qn(norm_dim_, norm_dim_, 0.0), Pmat(j+2, j+1, 0.0);
    ublas::matrix<double> Qn(j+1, norm_dim_, 0.0), Qt(j+1, tang_dim_, 0.0);
    for (int k = 0; k < norm_dim_; k++) QT_Qn(k,k) = 1.0;
    for (int l = 0; l < j+1; l++) {
      for (int k = 0; k < j+2; k++)
        Pmat(k,l) = P(l*(j+2)+k); // only first ncol columns of P are returned
      for (int k = 0; k < norm_dim_; k++)
        Qn(l,k) = QT(k + l*(j+1));
      for (int k = 0; k < tang_dim_; k++)
        Qt(l,k) = QT(k + norm_dim_ + l*(j+1));
    }

#ifdef DEBUG
    { // Test that Qn^T*Qn = I and Qt^T*Qt = I      
      ublas::matrix<double> In(norm_dim_, norm_dim_, 0.0);
      In = ublas::prod(ublas::trans(Qn), Qn);
      for (int i = 0; i < norm_dim_; i++)
        for (int j = 0; j < norm_dim_; j++)
          if ( ( (i == j) && (fabs(In(i,j)-1.0) > 1e-12) ) ||
               ( (i != j) && (fabs(In(i,j)) > 1e-12) ) ) {
            cerr << "TRISQP: Qn problem" << endl;
            cerr << "TRISQP: In(" << i << "," << j << ") = " << In(i,j) << endl;
            throw(-1);
          }
      ublas::matrix<double> It(tang_dim_, tang_dim_, 0.0);
      It = ublas::prod(ublas::trans(Qt), Qt);
      for (int i = 0; i < tang_dim_; i++)
        for (int j = 0; j < tang_dim_; j++)
          if ( ( (i == j) && (fabs(It(i,j)-1.0) > 1e-12) ) ||
               ( (i != j) && (fabs(It(i,j)) > 1e-12) ) ) {
            cerr << "TRISQP: Qt problem" << endl;
            throw(-1);
          }
    }
#endif
#if 0
    for (int k = 0; k <= j+1; k++) {
      cout << "C_(" << k << ",:) = ";
      for (int l = 0; l <= j; l++)
        cout << C_(k,l) << " ";
      cout << endl;
    }
#endif
    // solve for the normal step
    double nu_ls = solveLeastSquaresOverSphere(j+2, norm_dim_, norm_radius,
                                               Sigma, Pmat, QT_Qn, f_, yn_);
    cout << "> normal step norm = " << norm_2(yn_) << endl;
    
    //if (nu_ls > kEpsilon) active = true;
    if (nu_ls > kEpsilon) {
      cout << "LS normal step trust region is active !!!!!!!!!!!!" << endl;
      cout << "nu_ls = " << nu_ls << ": norm_2(yn_) = " << norm_2(yn_) << endl;
    }
    
    // The following vector and matrix ranges permit easier manipulation
    ublas::vector_range<ublas::vector<double> >
        g_r(g_, ublas::range(0,j+2)),
        f_r(f_, ublas::range(0,j+2)),
        y_r(y_, ublas::range(0,j+1)),
        yn_r(yn_, ublas::range(0,norm_dim_)),
        yt_r(yt_, ublas::range(0,tang_dim_)),
        sig_r(sig_, ublas::range(0,j+1));
    ublas::matrix_range<ublas::matrix<double> >
        B_r(B_, ublas::range(0,j+2), ublas::range(0,j+1)),
        C_r(C_, ublas::range(0,j+2), ublas::range(0,j+1)),
        Vprods_r(Vprods_, ublas::range(0,j+1), ublas::range(0,j+2)),
        Lambdaprods_r(Lambdaprods_, ublas::range(0,j+1), ublas::range(0,j+2));
    
    // BTilde is needed for both the tangential step computation, and, later,
    // for the predicted decrease
    ublas::matrix<double> BTilde = ublas::prod(Vprods_r, B_r);

#if 0
    for (int k = 0; k <= j; k++) {
      cout << "Vprods_(" << k << ",:) = ";
      for (int l = 0; l <= j+1; l++)
        cout << Vprods_r(k,l) << " ";
      cout << endl;
    }
    for (int k = 0; k <= j+1; k++) {
      cout << "B(" << k << ",:) = ";
      for (int l = 0; l <= j; l++)
        cout << B_r(k,l) << " ";
      cout << endl;
    }
#endif
    
    // When including A^T in Arnoldi
    ublas::matrix<double> CTilde = ublas::prod(Lambdaprods_r, C_r);
    BTilde -= ublas::trans(CTilde);
        
    // TEMP: try adding I to H
    //for (int k = 0; k <= j; k++) BTilde(k,k) -= 1.0;

#if 0
    for (int k = 0; k <= j; k++) {
      cout << "BTilde(" << k << ",:) = ";
      for (int l = 0; l <= j; l++)
        cout << BTilde(k,l) << " ";
      cout << endl;
    }
#endif
    
    // solve for the tangential step, if necessary
    old_nu = nu;
    nu = 0.0;
    active = false;
    if (tang_dim_ > 0) {
      // compute the Hessian for the tangential problem
      ublas::matrix<double> BTildeQt = ublas::prod(BTilde, Qt);
      ublas::matrix<double> Hess = ublas::prod(ublas::trans(Qt), BTildeQt);
      
      // compute the gradient for the tangential problem
      ublas::vector<double> Qnyn = ublas::prod(Qn, yn_r);
      ublas::vector<double> BTildeQnyn = ublas::prod(BTilde, Qnyn);
      for (int k = 0; k <= j; k++)
        BTildeQnyn(k) -= g_(0)*Vprods_(k,0); // the neg sign because b is on rhs
      ublas::vector<double> gt = ublas::prod(ublas::trans(Qt), BTildeQnyn);
#if 1
      for (int k = 0; k < tang_dim_; k++)
        cout << "gt(" << k << ") = " << gt(k) << endl;
#endif 
      // solve the tangential problem    
      double tang_radius = sqrt(radius*radius - inner_prod(yn_r, yn_r));
      //old_nu = nu; // moved up top
#if 0
      if (norm_2(gt) < kEpsilon) {
        for (int k = 0; k < tang_dim_; k++) yt_(k) = 0.0;
        nu = 0.0;
      } else {
        solveTrustReduced(tang_dim_, Hess, tang_radius, gt, yt_, nu, pred);
      }
#endif
      solveTrustReduced(tang_dim_, Hess, tang_radius, gt, yt_, nu, pred);
      if (nu > kEpsilon) active = true;      
      //cout << "TRISQP: After solveTrustReduced..." << endl;
      //cout << "\tpred = " << pred << endl;
#if 0
      ublas::matrix<double> QT_Qt(tang_dim_, tang_dim_, 0.0);
      for (int k = 0; k < tang_dim_; k++) QT_Qt(k,k) = 1.0;
      ublas::vector<double> y_norm = ublas::prod(Qn, yn_r);
      ublas::vector<double> res_red = -ublas::prod(B_r, y_norm) + g_r; 
      double nu_ls = solveLeastSquaresOverSphere(j+2, tang_dim_, tang_radius,
                                                 Sigma, Pmat, QT_Qt, res_red, yt_);
#endif
    }

    // form reduced-space solution (y, y_tang, and y_norm)
    y_tang.resize(j+1, 0.0);
    ublas::vector<double> y_norm(j+1, 0.0);    
    y_norm = ublas::prod(Qn, yn_r);
    y_r = y_norm;
    if (tang_dim_ > 0) {
      y_tang = ublas::prod(Qt, yt_r);  
      y_r += y_tang;
    }
    //cout << "TRISQP: norm_2(y_r) = " << norm_2(y_r) << endl;
    
    // solve for Lagrange multipliers
    ublas::vector<double> res_red = ublas::prod(B_r, y_r) - g_r; //neg
    ublas::matrix<double> CTilde_trans = ublas::trans(CTilde);
    ublas::vector<double> rhs = -ublas::prod(Vprods_r, res_red);

    // When including A^T in Arnoldi (effectively constructs BTilde*y_r)
    rhs += ublas::prod(CTilde_trans, y_r);

    // TEMP: try adding I to H
    //rhs += y_r;
    //if (tang_dim_ > 0) rhs -= nu*y_tang;

    if (norm_dim_ == 0) {
      cerr << "TRISQP: norm_dim_ = 0" << endl;
      throw(-1);
    }
      
    ublas::matrix<double> CTildeQn = ublas::prod(CTilde, Qn);
    ublas::vector<double> mult_rhs = ublas::prod(ublas::trans(Qn), rhs);
    ublas::vector<double> Sigma_mult;
    //computeSVD(j+1, j+1, CTilde_trans, Sigma_mult, P, QT);
    computeSVD(j+1, norm_dim_, CTildeQn, Sigma_mult, P, QT);
    cout << "TRISQP: after computeSVD(CTildeQn)..." << endl;
    ublas::matrix<double> QTmat(j+1, j+1, 0.0);
    Pmat.resize(j+1, norm_dim_);
    for (int l = 0; l < norm_dim_; l++) {
      for (int k = 0; k <= j; k++)
        Pmat(k,l) = P(l*(j+1)+k); // only first ncol columns of P are returned
      for (int k = 0; k < norm_dim_; k++)
        QTmat(k,l) = QT(l*(norm_dim_)+k);
    }
#if 0
    double lam_radius = 1e+100;
    solveLeastSquaresOverSphere(j+1, j+1, lam_radius, Sigma_mult, Pmat, QTmat,
                                rhs, sig_);
#endif
    cout << "TRISQP: multiplier matrix condition number = "
         << Sigma_mult(0)/Sigma_mult(norm_dim_-1) << endl;
    solveUnderdeterminedMinNorm(norm_dim_, j+1, Sigma_mult, ublas::trans(QTmat),
                                ublas::trans(Pmat), mult_rhs, sig_);
    cout << "TRISQP: norm of multipliers = " << norm_2(sig_r) << endl;
    
    // compute predicted decrease in the Augmented Lagrangian and its parameter
    ublas::vector<double> res_proj(j+1, 0.0);
    for (int k = 0; k <= j; k++) res_proj(k) = -g_(0)*Vprods_(k,0);
    res_proj += 0.5*ublas::prod(BTilde, y_r);
    double pred_opt = -ublas::inner_prod(res_proj, y_r);
    res_proj = ublas::prod(CTilde, y_r);
    for (int k = 0; k <= j; k++) res_proj(k) -= f_(0)*Lambdaprods_(k,0);
    pred_opt -= ublas::inner_prod(sig_r, res_proj);
    res_red = ublas::prod(C_r, y_r) - f_r;
    double pred_feas = -ublas::inner_prod(res_red, res_red);
    pred_feas += ublas::inner_prod(f_r, f_r); // = f_(0)*f_(0);
    const double rho_mu = 0.01;
    if (pred_feas > kEpsilon) // in case constraints are satisfied
      mu = std::max(mu, -2.0*pred_opt/(pred_feas*(1.0 - rho_mu)));
    pred = pred_opt + 0.5*mu*pred_feas;

    // This is the residual norm of the constraint equation
    gamma = norm_2(res_red);
    
    // build the primal residual
    x.dual() = 0.0;
    for (int k = 0; k <= j; k++)
      x.dual().EqualsAXPlusBY(1.0, x.dual(), sig_(k) - y_(k), LambdaTilde_[k]);
    x.primal() = 0.0;
    mat_vec(x, r_); // r += A^T* LambdaTilde*(sig - y)
    if (tang_dim_ > 0) {
      for (int k = 0; k <= j; k++) {
        // r += nu*I*pt
        r_.primal().EqualsAXPlusBY(1.0, r_.primal(), nu*y_tang(k), VTilde_[k]);
      }
    }
    res_red = ublas::prod(B_r, y_r) - g_r; // grad on rhs
    for (int k = 0; k <= j+1; k++) {
      // r += grad(L) + H(pn + pt)
      r_.primal().EqualsAXPlusBY(1.0, r_.primal(), res_red(k), V_[k]);
    }
    beta = r_.primal().Norm2();

    // build residual of constraint
    r_.dual() = 0.0;
    for (int k = 0; k <= j+1; k++) {
      // r = A*p + c
      r_.dual().EqualsAXPlusBY(1.0, r_.dual(), res_red(k), Lambda_[k]);
    }
    
#if 0
    // TEMP: confirm VTilde is orthonormal
    cout << std::setprecision(3);
    for (int k = 0; k <= j; k++) {
      cout << "V_[" << k << "]^T * V_[:] = ";
      for (int l = 0; l <= j; l++) {
        double tmp = InnerProd(VTilde_[k], VTilde_[l]);
        cout << tmp << " ";
      }
      cout << endl;
    }
#endif

    cout << "TRISQP: sing. values = ";
    for (int k = 0; k < j+1; k++)
      cout << Sigma(k) << " ";
    cout << endl;
    
    // compute norm of A*pn = A*VTilde*Qn*yn, which measures quality of null spc
    double null_norm = 0.0;
    for (int k = 0; k < tang_dim_; k++)
      null_norm += Sigma(k+norm_dim_)*Sigma(k+norm_dim_)*yt_(k)*yt_(k);
    null_norm = sqrt(null_norm);
    if (norm_2(yt_) < kEpsilon)
      null_norm = 1.0;
    else
      null_norm /= norm_2(yt_);

    // output
    if (j == 0) {
      boost::format col_head(string("%|6| %|8t| %|-12| %|20t| %|-12|")+
                             string(" %|32t| %|-12| %|44t| %|-12|")+
                             string("%|56t| %|-12| %|68t| %|-12|\n"));
      col_head % "# iter" % "grad. res." % "rel. feas." % "null qual."
          % "norm_dim" % "tang_dim" % "lambda";
      writeKrylovHeader(fout, "TRISQP", grad_tol, grad_norm0, col_head);
      //grad_norm0 = beta;
    }
    fout << boost::format(string("%|6| %|8t| %|-12.6| %|20t| %|-12.6|")+
                          string(" %|32t| %|-12.6| %|44t| %|-12|")+
                          string(" %|56t| %|-12| %|68t| %|-12.6|\n"))
        % (j+1) % (beta/grad_norm0) % (gamma/feas_norm0)
        % null_norm % norm_dim_ % tang_dim_ % nu;
    fout.flush();
    
    // check for convergence
    bool converged = true;
    if (j == 0) converged = false;
    if (active) {
      if (null_norm > feas_tol)
        converged = false; // trust-region active, but null space not good
#if 0
      if (beta > grad_norm0*grad_tol)
        converged = false; // optimality tol not met
#endif
    } else {
      //cerr << "TODO: TRISQP convergence not working as expected" << endl;
      //throw(-1);
      if (gamma > feas_norm0*feas_tol)
        converged = false; // feasibility tol not met, despite room to improve
      if (beta > grad_norm0*grad_tol)
        converged = false; // optimality tol not met, despite room to improve
    }
    
    if (converged) break;
  }

  // build the primal and dual parts of the solution
  j = norm_dim_ + tang_dim_;
  x.primal() = 0.0;
  for (int k = 0; k < j; k++)
    x.primal().EqualsAXPlusBY(1.0, x.primal(), y_(k), VTilde_[k]);
  x.dual() = 0.0;
  for (int k = 0; k < j; k++)
    x.dual().EqualsAXPlusBY(1.0, x.dual(), sig_(k), LambdaTilde_[k]);
  
  if (check) {
    // check the optimality and feasibility
    mat_vec(x, r_);
    r_.primal() -= b.primal();
    r_.dual() -= b.dual();
    if (tang_dim_ > 0)
      for (int k = 0; k < j; k++)
        r_.primal().EqualsAXPlusBY(1.0, r_.primal(), nu*y_tang(k), VTilde_[k]);
    double res = r_.primal().Norm2();
    fout << "# TRISQP final (true) grad. norm : |grad|/|grad0| = "
         << res/grad_norm0 << endl;    
    fout << "# TRISQP final (true) feas. norm : |feas|/|feas0| = "
         << r_.dual().Norm2()/feas_norm0 << endl;    
    // check that solution satisfies the trust region
    double x_norm = x.primal().Norm2();
    if (fabs(nu) < kEpsilon) {
      if (x_norm > radius) {
        cerr << "TRISQP: lambda = 0 and solution is not inside trust region."
             << endl;
        throw(-1);
      }
    } else { // solution should lie on trust region
      if (x_norm - radius > 1.e-6) {
        cerr << "TRISQP: lambda > 0 and solution is not on trust region."
             << endl;
        cerr << "x_norm - radius = " << x_norm - radius << endl;     
        throw(-1);
      }
    }
  }
}
#endif
// ==============================================================================
//class ReducedKKTProduct;
template <class Vec, class PrimVec, class DualVec>
void TRISQP<Vec, PrimVec, DualVec>::Solve(
    double grad_tol, double feas_tol, double radius, const Vec & b, Vec & x,
    MatrixVectorProduct<Vec> & mat_vec, Preconditioner<Vec> & precond,
    int & iters, double & mu, double & pred_opt, double & pred_feas,
    bool & active, ostream & fout, const bool & check, const bool & dynamic) {
  if (maxiter_ < 1) {
    cerr << "TRISQP::Solve(): SubspaceSize must be called first." << endl;
    throw(-1);
  }
  if ( (grad_tol < 0.0) || (feas_tol < 0.0) ) {
    cerr << "TRISQP::Solve(): tolerances must be positive, grad_tol = "
         << grad_tol << ": feas_tol = " << feas_tol << endl;
    throw(-1);
  }
  if (radius <= 0.0) {
    cerr << "TRISQP::Solve(): trust-region radius must be positive, radius = "
         << radius << endl;
    throw(-1);
  }

  // clear some vectors and matrices
  y_.clear();
  sig_.clear();
  B_.clear();
  C_.clear();
  D_.clear();
  Vprods_.clear();
  VZprods_.clear();
  
  // The variable names below try to follow the paper as closely as possible,
  // although some discrepancies are inevitable
  r_ = b;
  V_.push_back(b.primal());
  iters = 0;
  
  // calculate the norm of the rhs vector and initialize the RHS of the reduced
  // system
  double grad_norm0 = b.primal().Norm2();
  double feas_norm0 = b.dual().Norm2();
  double beta = grad_norm0;
  double gamma = feas_norm0;
  V_[0] /= beta; // note, V_[0] is set to b.primal() when initialized
  g_.clear();
  g_(0) = beta;
  
  f_.clear();
  Lambda_.push_back(b.dual());
  if (feas_tol >= 1.0) {
    // the constraints are satisfied, so assume that b.dual() = 0.0 and we need
    // to find an alternative to Lambda_[0]
    Lambda_[0] = 1.0;
    double tmp = Lambda_[0].Norm2();
    Lambda_[0] /= tmp;
  } else {
    Lambda_[0] /= gamma;
    f_(0) = gamma;
  }

  // initialize Z_[0]
  r_.primal() = 0.0;
  r_.dual() = Lambda_[0];
  mat_vec(r_, x);
  Z_.push_back(x.primal());
  D_(0,0) = Z_[0].Norm2();
  Z_[0] /= D_(0,0);

  double red_mu = mu; //0.1*feas_tol/grad_tol;
  
  // nu is the Lagrange multiplier for the tangential-step subproblem
  double nu = 0.0;
  double old_nu = 0.0;
  ublas::vector<double> res_red_old(1, -gamma);
  bool V_lin_dep = false;
  bool Lambda_lin_dep = false;
  int j = 0;
  active = false;
  for (j = 0; j < maxiter_; j++) {
    iters++;

    // preconditioning
    precond.set_diagonal(std::max(nu + (nu-old_nu),0.0));
    if (j == 0) {
      VTilde_.push_back(V_[j]);
    } else {
      // use x to store result of preconditioning; restarting not possible
      //r_.primal() = V_[j];
      //r_.dual() = Lambda_[j];
      precond(r_, x);
      VTilde_.push_back(x.primal());
    }
    
    cout << "TRISQP: after preconditioning" << endl;
    
    // Orthonormalize VTilde_[j] against previous vectors
    try {
      modGramSchmidt(j-1, VTilde_);
    } catch (string err_msg) {
      cerr << "TRISQP: VTilde vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      V_lin_dep = depend;
    }
    
    // Expand the subspace for V_ and Lambda_
    r_.primal() = VTilde_[j];
    r_.dual() = Lambda_[j];
    mat_vec(r_, x);
    V_.push_back(x.primal());
    Lambda_.push_back(x.dual());
    cout << "TRISQP: after first matrix-vector product..." << endl;
    
    // Orthogonalize V_[j+1] and Lambda_[j+1] against previous vectors, and
    // compute the matrices B_ and C_
    try {
      modGramSchmidt(j, C_, Lambda_);
    } catch (string err_msg) {
      cerr << "TRISQP: Lambda vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP: Lambda linear dependence" << endl;
      Lambda_lin_dep = depend;
    }
    // if V_[j+1] is small, and Lambda_[j+1] is linearly dependent, it is
    // likekly that the Hessian of the Lagrangian has zero eigenvalues; break
    if (Lambda_lin_dep) 
      if (V_[j+1].Norm2() < kEpsilon) {
        cout << "TRISQP: Lambda_lin_dep and V_ = 0" << endl;
        VTilde_.pop_back();
        V_.pop_back();
        Lambda_.pop_back();
        --iters;
        break;
      }
    try {
      modGramSchmidt(j, B_, V_);
    } catch (string err_msg) {
      cerr << "TRISQP: V vector orthogonalization: " << err_msg << endl;
      throw(-1);
    } catch (bool depend) {
      cout << "TRISQP: V linear dependence" << endl;
      V_lin_dep = depend;
    }    
    cout << "TRISQP: after Gram-Schmidt on V and Lambda..." << endl;

    // add new row and column to matrices V_prods = VTilde^T V and
    // VZ_prods = VTilde^T Z
    for (int k = 0; k < j+1; k++) {
      Vprods_(j,k) = InnerProd(VTilde_[j], V_[k]);
      Vprods_(k,j+1) = InnerProd(VTilde_[k], V_[j+1]);
      VZprods_(j,k) = InnerProd(VTilde_[j], Z_[k]);
      VZprods_(k,j) = InnerProd(VTilde_[k], Z_[j]);
    }
    
    // expand subspace for Z_
    r_.primal() = 0.0;
    r_.dual() = Lambda_[j+1];
    mat_vec(r_, x);
    Z_.push_back(x.primal());
    cout << "TRISQP: after second matrix-vector product..." << endl;

    // orthogonalize the Z_
    // NOTE: this is not robust, and modGramSchmidt should be adapted
    for (int k = 0; k < j+1; k++) {      
      D_(k,j+1) = InnerProd(Z_[k], Z_[j+1]);
      Z_[j+1].EqualsAXPlusBY(1.0, Z_[j+1], -D_(k,j+1), Z_[k]);
    }
    D_(j+1,j+1) = Z_[j+1].Norm2();
    if (D_(j+1,j+1) > kEpsilon) Z_[j+1] /= D_(j+1,j+1);

#if 0
    double red_mu = 0.01; //feas_tol/grad_tol; //gamma/beta; //  //feas_norm0/grad_norm0; //
    mu = red_mu; // TEMP
#endif

    //red_mu = (gamma/feas_norm0)/(beta/grad_norm0);
    
    // The following vector and matrix ranges permit easier manipulation
    ublas::vector_range<ublas::vector<double> >
        y_r(y_, ublas::range(0,j+1)),
        g_r(g_, ublas::range(0,j+2)),
        f_r(f_, ublas::range(0,j+2)),
        sig_r(sig_, ublas::range(0,j+1));
    ublas::matrix_range<ublas::matrix<double> >
        B_r(B_, ublas::range(0,j+2), ublas::range(0,j+1)),
        C_r(C_, ublas::range(0,j+2), ublas::range(0,j+1)),
        C_sqr(C_, ublas::range(0,j+1), ublas::range(0,j+1)),
        D_r(D_, ublas::range(0,j+1), ublas::range(0,j+1)),
        Vprods_r(Vprods_, ublas::range(0,j+1), ublas::range(0,j+2)),
        VZprods_r(VZprods_, ublas::range(0,j+1), ublas::range(0,j+1));

    if (!active)
      sig_r += red_mu*res_red_old;
    
    // evaluate the reduced-space system matrix
#if 0
    ublas::matrix<double> A = ublas::prod(Vprods_r, B_r) - ublas::trans(C_sqr)
        + red_mu*ublas::prod(ublas::trans(C_r), C_r);
#endif

    cout << "Why does this work, but the above does not?" << endl;
    //throw(-1);
    ublas::matrix<double> A = ublas::prod(Vprods_r, B_r)
        - ublas::prod(VZprods_r, D_r)
        + red_mu*ublas::prod(ublas::trans(C_r), C_r);
    
    cout << "TRISQP: after constructing A..." << endl;

#if 0
    cout << "A - AT= " << endl;
    for (int k = 0; k < j+1; k++) {
      for (int l = 0; l < j+1; l++)
        cout << A(k,l) - A(l,k) << " ";
      cout << endl;
    }

    cout << "C_sqr^{T} = " << endl;
    for (int k = 0; k < j+1; k++) {
      for (int l = 0; l < j+1; l++)
        cout << C_sqr(l,k) << " ";
      cout << endl;
    }
#endif
    
    // evaluate the reduced-space rhs
    ublas::vector<double> rhs = -grad_norm0*ublas::column(Vprods_r, 0);
    rhs += ublas::prod(ublas::trans(C_sqr), sig_r);
    //ublas::vector<double> D_sig = ublas::prod(D_r, sig_r);
    //rhs += ublas::prod(VZprods_r, D_sig);

    rhs -= red_mu*feas_norm0*ublas::trans(ublas::row(C_r, 0));
    cout << "TRISQP: after constructing rhs..." << endl;

    double pred;
    solveTrustReduced(j+1, A, radius, rhs, y_, nu, pred);
    if (nu > kEpsilon) active = true;
    pred += feas_norm0*sig_(0); // correct pred with -c^{T} * psi 
    
    
    // the constraint (dual) residual
    ublas::vector<double> res_red = ublas::prod(C_r, y_r);
    res_red(0) -= feas_norm0;
    gamma = norm_2(res_red);
    
    // build the residual
    r_ = 0.0;
    ublas::vector<double> Dsig_minus_Dy = ublas::prod(D_r, sig_r - y_r);
    for (int k = 0; k < j+1; k++) {
      // r += nu*I*p
      r_.primal().EqualsAXPlusBY(1.0, r_.primal(), nu*y_(k), VTilde_[k]);
      // r += A^T * psi - A^T Lambda y
      r_.primal().EqualsAXPlusBY(1.0, r_.primal(), Dsig_minus_Dy(k), Z_[k]);
    }
    ublas::vector<double> opt_red = ublas::prod(B_r, y_r) - g_r; // grad on rhs
    ublas::vector<double> muDres = red_mu*ublas::prod(
        ublas::project(D_, ublas::range(0, j+2), ublas::range(0, j+2)),res_red);
    for (int k = 0; k < j+2; k++) {
      // r += grad(L) + H*p
      r_.primal().EqualsAXPlusBY(1.0, r_.primal(), opt_red(k), V_[k]);
      // c += c + A*p
      r_.dual().EqualsAXPlusBY(1.0, r_.dual(), res_red(k), Lambda_[k]);
      // r += mu*A^T * (c + A*p)
      r_.primal().EqualsAXPlusBY(1.0, r_.primal(), muDres(k), Z_[k]); 
    }
    beta = r_.primal().Norm2();
    if (fabs(r_.dual().Norm2() - gamma) > 1e-6) {
      cerr << "TRISQP::Solve(): reduced dual residual does not match true"
           << endl;
      throw(-1);
    }

    // update pred_feas and pred_opt
    pred_feas = -ublas::inner_prod(res_red, res_red);
    pred_feas += ublas::inner_prod(f_r, f_r); // = f_(0)*f_(0);
    pred_opt = pred - 0.5*red_mu*pred_feas;

#if 0
    // update the merit penalty parameter
    const double rho_mu = 0.01;
    if (pred_feas > kEpsilon) // in case constraints are satisfied
      mu = std::max(mu, -2.0*pred_opt/(pred_feas*(1.0 - rho_mu)));
#endif
    
    // output
    if (j == 0) {
      boost::format col_head(string("%|6| %|8t| %|-12| %|20t| %|-12|")+
                             string(" %|32t| %|-12|\n"));
      col_head % "# iter" % "grad. res." % "rel. feas." % "lambda";
      writeKrylovHeader(fout, "TRISQP", grad_tol, grad_norm0, col_head);
      //grad_norm0 = beta;
    }
    fout << boost::format(string("%|6| %|8t| %|-12.6| %|20t| %|-12.6|")+
                          string(" %|32t| %|-12.6|\n"))
        % (j+1) % (beta/grad_norm0) % (gamma/feas_norm0) % nu;
    fout.flush();
    
    // check for convergence
    // WARNING: not general yet
    bool converged = true;
    if (j == 0) converged = false;
    if (beta > grad_norm0*grad_tol)
      converged = false; // optimality tol not met
    if ( (!active) && (gamma > feas_norm0*feas_tol) )
      converged = false; // feasibility tol not met despite room to improve

    if (j == maxiter_-1) break;
    if (converged) break;

#if 0
    // Find the reduced-space Lagrange multipliers
    if ( (!active) && (j < maxiter_-1))
      ublas::project(sig_, ublas::range(0, j+2)) += red_mu*res_red;
    if ( (active) && (j < maxiter_-1) ) sig_.clear();
    cout << "TRISQP: after solving for multipliers..." << endl;
    cout << "\tnorm of multipliers = " << norm_2(sig_) << endl;
#endif
    res_red_old.resize(j+2);
    res_red_old = res_red;

  }
  j = iters; // breaking may make j inconsistent 
    
  // build the primal and dual parts of the solution
  x.primal() = 0.0;
  for (int k = 0; k < j; k++)
    x.primal().EqualsAXPlusBY(1.0, x.primal(), y_(k), VTilde_[k]);
  x.dual() = 0.0;
  for (int k = 0; k < j; k++)
    x.dual().EqualsAXPlusBY(1.0, x.dual(), sig_(k), Lambda_[k]);
  
  if (check) {
    // check the feasibility
    mat_vec(x, r_);
    r_.dual() -= b.dual();
    fout << "# TRISQP final (true) feas. norm : |feas|/|feas0| = "
         << r_.dual().Norm2()/feas_norm0 << endl;

    // check the optimality
    // temporarily add mu*(Ap + c) to psi
    x.dual().EqualsAXPlusBY(1.0, x.dual(), red_mu, r_.dual());
    mat_vec(x, r_);
    r_.dual() -= b.dual();
    r_.primal() -= b.primal();
    for (int k = 0; k < j; k++)
      r_.primal().EqualsAXPlusBY(1.0, r_.primal(), nu*y_(k), VTilde_[k]);
    fout << "# TRISQP final (true) grad. norm : |grad|/|grad0| = "
         << r_.primal().Norm2()/grad_norm0 << endl;
    x.dual().EqualsAXPlusBY(1.0, x.dual(), -red_mu, r_.dual());
    
    // check that solution satisfies the trust region
    double x_norm = x.primal().Norm2();
    if (fabs(nu) < kEpsilon) {
      if (x_norm > radius) {
        cerr << "TRISQP: lambda = 0 and solution is not inside trust region."
             << endl;
        throw(-1);
      }
    } else { // solution should lie on trust region
      if (x_norm - radius > 1.e-6*radius) {
        cerr << "TRISQP: lambda > 0 and solution is not on trust region."
             << endl;
        cerr << "x_norm - radius = " << x_norm - radius << endl;     
        throw(-1);
      }
    }
  }
}
// ==============================================================================
template <class Vec, class PrimVec, class DualVec>
void TRISQP<Vec, PrimVec, DualVec>::Correct2ndOrder(
    const DualVec & ceq, PrimVec & x, double radius, double & mu,
    double & pred_opt, double & pred_feas) {
  // compute the minimum norm solution

  int j = VTilde_.size();
  ublas::vector_range<ublas::vector<double> >
      y_r(y_, ublas::range(0,j)),
      g_r(g_, ublas::range(0,j+1)),
      f_r(f_, ublas::range(0,j+1)),
      sig_r(sig_, ublas::range(0,j));
  ublas::matrix_range<ublas::matrix<double> >
      B_r(B_, ublas::range(0,j+1), ublas::range(0,j)),
      C_r(C_, ublas::range(0,j+1), ublas::range(0,j)),
      C_sqr(C_, ublas::range(0,j), ublas::range(0,j)),
      D_r(D_, ublas::range(0,j), ublas::range(0,j)),
      Vprods_r(Vprods_, ublas::range(0,j), ublas::range(0,j+1)),
      VZprods_r(VZprods_, ublas::range(0,j), ublas::range(0,j));

  cout << "TRISQP::Correct2ndOrder(): after setting proxies..." << endl;
  
  // find the LHS matrix
  ublas::matrix<double> DTD = ublas::prod(ublas::trans(D_r),D_r);

  // find the RHS vector = Lambda^T * ceq
  ublas::vector<double> rhs(j, 0.0);
  for (int k = 0; k < j; k++)
    rhs(k) = InnerProd(Lambda_[k], ceq);

  cout << "TRISQP::Correct2ndOrder(): after forming lhs and rhs..." << endl;

  cout << "DTD = " << endl;
  for (int k = 0; k < j; k++) {
    for (int l = 0; l < j; l++)
      cout << DTD(k,l) << " ";
    cout << endl;
  }   
  cout << endl;
  
  // solve for sig = (DTD)^{-1} rhs
  ublas::vector<double> Sigma, P, QT, sig_cor(j);
  computeSVD(j, j, DTD, Sigma, P, QT);
  // move QT and P into ublas::matrix format
  ublas::matrix<double> Pmat(j, j, 0.0), QTmat(j, j, 0.0);
  for (int l = 0; l < j; l++) {
    for (int k = 0; k < j; k++) {
      Pmat(k,l) = P(l*j+k); 
      QTmat(k,l) = QT(l*j+k);
    }
  }
  solveUnderdeterminedMinNorm(j, j, Sigma, Pmat, QTmat, rhs, sig_cor);
  
  cout << "TRISQP::Correct2ndOrder(): after solving for sig_cor..." << endl;

#if 0
  // find normal step: VTilde y, where (Z^T VTilde)y = -D*sig_cor
  
  // compute SVD of VTilde^T Z: this is necessary, because Z may not be full
  // rank, in general
  computeSVD(j, j, VZprods_, Sigma, P, QT);
  // move QT and P into ublas::matrix format
  ublas::matrix<double> U(j, j, 0.0), VT(j, j, 0.0);
  for (int l = 0; l < j; l++) {
    for (int k = 0; k < j; k++) {
      U(l,k) = QT(l*j+k); // U = (Q^T)^T = Q
      VT(l,k) = P(l*j+k); // VT = P^T
    }
  }
  rhs = -ublas::prod(D_r, sig_cor);
  ublas::vector<double> y_cor(j);
  solveUnderdeterminedMinNorm(j, j, Sigma, U, VT, rhs, y_cor);
#endif

  // find normal step y = - (VTilde^T Z) D*sig_cor
  ublas::vector<double> tmp = ublas::prod(D_r, sig_cor);
  ublas::vector<double> y_cor = ublas::prod(VZprods_r, tmp);
  
  cout << "TRISQP::Correct2ndOrder(): after solving for y_cor..." << endl;
  
  // clip y at trust radius, if necessary
  double norm_y = norm_2(y_cor);
  if (norm_y > radius)
    y_cor *= (radius/norm_y);
  
  // Add second-order correction to x
  for (int k = 0; k < j; k++)
    x.EqualsAXPlusBY(1.0, x, y_cor(k), VTilde_[k]);

  // Correct merit function
  ublas::matrix<double> BTilde = ublas::prod(Vprods_r, B_r)
      - ublas::prod(VZprods_r, D_r);
  ublas::matrix<double> CTC = ublas::prod(ublas::trans(C_r), C_r);
  // pred -= y_cor^T(VTilde^T v_1 beta)
  pred_opt -= -g_(0)*inner_prod(ublas::trans(y_cor), ublas::column(Vprods_r, 0));
  // pred -= 0.5 y_cor^T BTilde y_cor
  pred_opt -= 0.5*inner_prod(ublas::trans(y_cor), ublas::prod(BTilde, y_cor));
  // pred -= y_cor^T BTilde y_r
  pred_opt -= inner_prod(ublas::trans(y_cor), ublas::prod(BTilde, y_r));
  // pred -= y_cor^T (VTilde^T Z) sig_r
  pred_opt -= inner_prod(ublas::trans(y_cor), ublas::prod(VZprods_r, sig_r));
  
  // pred -= y_cor^T C^T C y_cor
  pred_feas -= inner_prod(ublas::trans(y_cor), ublas::prod(CTC, y_cor));
  // pred -= y_cor^T C^T (C^T y + c)
  ublas::vector<double> CT_res_red = ublas::prod(ublas::trans(C_r),
                                                 ublas::prod(C_r, y_r) - f_r);
  pred_feas -= 2.0*inner_prod(ublas::trans(y_cor), CT_res_red);
#if 0
  // update the merit penalty parameter
  const double rho_mu = 0.01;
  if (pred_feas > kEpsilon) // in case constraints are satisfied
    mu = std::max(mu, -2.0*pred_opt/(pred_feas*(1.0 - rho_mu)));    
#endif
  cout << "TRISQP::Correct2ndOrder(): after correcting pred..." << endl;
}
// ==============================================================================
#if 0
template <class Vec, class PrimVec, class DualVec>
void TRISQP<Vec, PrimVec, DualVec>::ReSolve(
    double grad_tol, double feas_tol, double radius, const Vec & b, Vec & x,
    MatrixVectorProduct<Vec> & mat_vec, Preconditioner<Vec> & precond,
    int & iters, double & mu, double & pred_opt, double & pred_feas,
    bool & active, ostream & fout, const bool & check, const bool & dynamic) {
  if ( (grad_tol < 0.0) || (feas_tol < 0.0) ) {
    cerr << "TRISQP::Solve(): tolerances must be positive, grad_tol = "
         << grad_tol << ": feas_tol = " << feas_tol << endl;
    throw(-1);
  }
  if (radius <= 0.0) {
    cerr << "TRISQP::Solve(): trust-region radius must be positive, radius = "
         << radius << endl;
    throw(-1);
  }

  int j = VTilde_.size() - 1;
  
  // calculate the norm of the rhs vector and initialize the RHS of the reduced
  // system
  double grad_norm0 = b.primal().Norm2();
  double feas_norm0 = b.dual().Norm2();

  // reform reduced-system initial gradient and constraint
  g_.clear();
  g_(0) = beta;
  f_.clear();
  if (feas_tol < 1.0) {
    f_(0) = gamma;
  }
  
  double red_mu = mu; //0.1*feas_tol/grad_tol;
  
  // nu is the Lagrange multiplier for the tangential-step subproblem
  double nu;
    
  // The following vector and matrix ranges permit easier manipulation
  ublas::vector_range<ublas::vector<double> >
      y_r(y_, ublas::range(0,j+1)),
      g_r(g_, ublas::range(0,j+2)),
      f_r(f_, ublas::range(0,j+2)),
      sig_r(sig_, ublas::range(0,j+1));
  ublas::matrix_range<ublas::matrix<double> >
      B_r(B_, ublas::range(0,j+2), ublas::range(0,j+1)),
      C_r(C_, ublas::range(0,j+2), ublas::range(0,j+1)),
      C_sqr(C_, ublas::range(0,j+1), ublas::range(0,j+1)),
      D_r(D_, ublas::range(0,j+1), ublas::range(0,j+1)),
      Vprods_r(Vprods_, ublas::range(0,j+1), ublas::range(0,j+2)),
      VZprods_r(VZprods_, ublas::range(0,j+1), ublas::range(0,j+1));
    
  // evaluate the reduced-space system matrix
  ublas::matrix<double> A = ublas::prod(Vprods_r, B_r)
      - ublas::prod(VZprods_r, D_r)
      + red_mu*ublas::prod(ublas::trans(C_r), C_r);
        
  // evaluate the reduced-space rhs
  ublas::vector<double> rhs = -grad_norm0*ublas::column(Vprods_r, 0);
  rhs += ublas::prod(ublas::trans(C_sqr), sig_r);
  rhs -= red_mu*feas_norm0*ublas::trans(ublas::row(C_r, 0));

  double pred;
  solveTrustReduced(j+1, A, radius, rhs, y_, nu, pred);
  if (nu > kEpsilon) active = true;
  pred += feas_norm0*sig_(0); // correct pred with -c^{T} * psi     
    
  // the constraint (dual) residual
  ublas::vector<double> res_red = ublas::prod(C_r, y_r);
  res_red(0) -= feas_norm0;
  gamma = norm_2(res_red);
    
  // build the residual
  r_ = 0.0;
  ublas::vector<double> Dsig_minus_Dy = ublas::prod(D_r, sig_r - y_r);
  for (int k = 0; k < j+1; k++) {
    // r += nu*I*p
    r_.primal().EqualsAXPlusBY(1.0, r_.primal(), nu*y_(k), VTilde_[k]);
    // r += A^T * psi - A^T Lambda y
    r_.primal().EqualsAXPlusBY(1.0, r_.primal(), Dsig_minus_Dy(k), Z_[k]);
  }
  ublas::vector<double> opt_red = ublas::prod(B_r, y_r) - g_r; // grad on rhs
  ublas::vector<double> muDres = red_mu*ublas::prod(
      ublas::project(D_, ublas::range(0, j+2), ublas::range(0, j+2)),res_red);
  for (int k = 0; k < j+2; k++) {
    // r += grad(L) + H*p
    r_.primal().EqualsAXPlusBY(1.0, r_.primal(), opt_red(k), V_[k]);
    // c += c + A*p
    r_.dual().EqualsAXPlusBY(1.0, r_.dual(), res_red(k), Lambda_[k]);
    // r += mu*A^T * (c + A*p)
    r_.primal().EqualsAXPlusBY(1.0, r_.primal(), muDres(k), Z_[k]); 
  }
  beta = r_.primal().Norm2();
  if (fabs(r_.dual().Norm2() - gamma) > 1e-6) {
    cerr << "TRISQP::Solve(): reduced dual residual does not match true"
         << endl;
    throw(-1);
  }

  // update pred_feas and pred_opt
  pred_feas = -ublas::inner_prod(res_red, res_red);
  pred_feas += ublas::inner_prod(f_r, f_r); // = f_(0)*f_(0);
  pred_opt = pred - 0.5*red_mu*pred_feas;

  // update the merit penalty parameter
  const double rho_mu = 0.01;
  if (pred_feas > kEpsilon) // in case constraints are satisfied
    mu = std::max(mu, -2.0*pred_opt/(pred_feas*(1.0 - rho_mu)));    
    
  // output
  fout << "# TRISQP: resolving at radius = " << radius << endl;
  fout << boost::format(string("%|6| %|8t| %|-12.6| %|20t| %|-12.6|")+
                        string(" %|32t| %|-12.6|\n"))
      % (j+1) % (beta/grad_norm0) % (gamma/feas_norm0) % nu;
  fout.flush();
    
  // build the primal and dual parts of the solution
  x.primal() = 0.0;
  for (int k = 0; k < j; k++)
    x.primal().EqualsAXPlusBY(1.0, x.primal(), y_(k), VTilde_[k]);
  x.dual() = 0.0;
  for (int k = 0; k < j; k++)
    x.dual().EqualsAXPlusBY(1.0, x.dual(), sig_(k), Lambda_[k]);
  
  if (check) {
    // check the feasibility
    mat_vec(x, r_);
    r_.dual() -= b.dual();
    fout << "# TRISQP final (true) feas. norm : |feas|/|feas0| = "
         << r_.dual().Norm2()/feas_norm0 << endl;

    // check the optimality
    // temporarily add mu*(Ap + c) to psi
    x.dual().EqualsAXPlusBY(1.0, x.dual(), red_mu, r_.dual());
    mat_vec(x, r_);
    r_.dual() -= b.dual();
    r_.primal() -= b.primal();
    for (int k = 0; k < j; k++)
      r_.primal().EqualsAXPlusBY(1.0, r_.primal(), nu*y_(k), VTilde_[k]);
    fout << "# TRISQP final (true) grad. norm : |grad|/|grad0| = "
         << r_.primal().Norm2()/grad_norm0 << endl;
    x.dual().EqualsAXPlusBY(1.0, x.dual(), -red_mu, r_.dual());
    
    // check that solution satisfies the trust region
    double x_norm = x.primal().Norm2();
    if (fabs(nu) < kEpsilon) {
      if (x_norm > radius) {
        cerr << "TRISQP: lambda = 0 and solution is not inside trust region."
             << endl;
        throw(-1);
      }
    } else { // solution should lie on trust region
      if (x_norm - radius > 1.e-6*radius) {
        cerr << "TRISQP: lambda > 0 and solution is not on trust region."
             << endl;
        cerr << "x_norm - radius = " << x_norm - radius << endl;     
        throw(-1);
      }
    }
  }
}
#endif
// ==============================================================================
#if 0
template <class Vec, class PrimVec, class DualVec>
void TRISQP<Vec, PrimVec, DualVec>::Correct2ndOrder(
    const DualVec & ceq, PrimVec & x, double radius, double & mu,
    double & pred) {

  int j = norm_dim_ + tang_dim_;
  
  // compute the SVD of C to find the null space approximation
  ublas::vector<double> Sigma, P, QT;
  computeSVD(j+1, j, C_, Sigma, P, QT);

#ifdef DEBUG
  // check the dimensions of the normal and tangential subspaces
#if 0
  int norm_dim_check = 0;
  for (int k = 0; k < (int)(j+1)/2; k++) {
    if (Sigma(k) < sqrt(kEpsilon)*Sigma(0)) break;  // what if no constraints?
    norm_dim_check++;
  }
  int tang_dim_check = j - norm_dim_check;
#else
  int norm_dim_check = 0;
  for (int k = 0; k < j; k++) {
    if (Sigma(k) < 0.01) break; //1e-2*Sigma(0)) break;  // what if no constraints?
    norm_dim_check++;
  }
  int tang_dim_check = j - norm_dim_check;
#endif
  
  if (norm_dim_check != norm_dim_) {
    cerr << "TRISQP::Correct2ndOrder(): norm_dim not compatible with C_ matrix."
         << endl;
    throw(-1);
  }
  if (tang_dim_check != tang_dim_) {
    cerr << "TRISQP::Correct2ndOrder(): tang_dim not compatible with C_ matrix."
         << endl;
    throw(-1);
  }
#endif

  // move QT and P into ublas::matrix format (to facilitate readability)
  ublas::matrix<double> QT_Qn(norm_dim_, norm_dim_, 0.0), Pmat(j+1, j, 0.0);
  ublas::matrix<double> Qn(j, norm_dim_, 0.0), Qt(j, tang_dim_, 0.0);
  for (int k = 0; k < norm_dim_; k++) QT_Qn(k,k) = 1.0;
  for (int l = 0; l < j; l++) {
    for (int k = 0; k < j+1; k++)
      Pmat(k,l) = P(l*(j+1)+k); // only first ncol columns of P are returned
    for (int k = 0; k < norm_dim_; k++)
      Qn(l,k) = QT(k + l*j);
    for (int k = 0; k < tang_dim_; k++)
      Qt(l,k) = QT(k + norm_dim_ + l*j);
  }

#ifdef DEBUG
  { // Test that Qn^T*Qn = I and Qt^T*Qt = I      
    ublas::matrix<double> In(norm_dim_, norm_dim_, 0.0);
    In = ublas::prod(ublas::trans(Qn), Qn);
    for (int i = 0; i < norm_dim_; i++)
      for (int j = 0; j < norm_dim_; j++)
        if ( ( (i == j) && (fabs(In(i,j)-1.0) > 1e-12) ) ||
             ( (i != j) && (fabs(In(i,j)) > 1e-12) ) ) {
          cerr << "TRISQP::Correct2ndOrder(): Qn problem" << endl;
          cerr << "\tIn(" << i << "," << j << ") = " << In(i,j) << endl;
          throw(-1);
        }
    ublas::matrix<double> It(tang_dim_, tang_dim_, 0.0);
    It = ublas::prod(ublas::trans(Qt), Qt);
    for (int i = 0; i < tang_dim_; i++)
      for (int j = 0; j < tang_dim_; j++)
        if ( ( (i == j) && (fabs(It(i,j)-1.0) > 1e-12) ) ||
             ( (i != j) && (fabs(It(i,j)) > 1e-12) ) ) {
          cerr << "TRISQP::Correct2ndOrder(): Qt problem" << endl;
          throw(-1);
        }
  }
#endif  

#if 0
  // project the constraint residual onto LambdaTilde
  f_.clear();
  for (int k = 0; k < j; k++)
    f_(k) = InnerProd(LambdaTilde_[k], ceq);
#endif

  // project the constraint residual onto Lambda and move to rhs
  ublas::vector<double> fhat(j+1, 0.0);
  for (int k = 0; k < j+1; k++)
    fhat(k) = -InnerProd(Lambda_[k], ceq);
    
  // solve for the normal step
  yn_.clear();
  double nu_ls = solveLeastSquaresOverSphere(j+1, norm_dim_, radius,
                                             Sigma, Pmat, QT_Qn, fhat, yn_);

  // Add 2nd-order correction to x
  //x.primal() = 0.0;
  ublas::vector_range<ublas::vector<double> >
      yn_r(yn_, ublas::range(0,norm_dim_));
  ublas::vector<double> y_norm = ublas::prod(Qn, yn_r);  
  for (int k = 0; k < j; k++)
    x.EqualsAXPlusBY(1.0, x, y_norm(k), VTilde_[k]);

  // The following vector and matrix ranges permit easier manipulation
  ublas::vector_range<ublas::vector<double> >
      g_r(g_, ublas::range(0,j+1)),
      f_r(f_, ublas::range(0,j+1)),
      y_r(y_, ublas::range(0,j)), // <-- needs updating
      sig_r(sig_, ublas::range(0,j));
  ublas::matrix_range<ublas::matrix<double> >
      B_r(B_, ublas::range(0,j+1), ublas::range(0,j)),
      C_r(C_, ublas::range(0,j+1), ublas::range(0,j)),
      Vprods_r(Vprods_, ublas::range(0,j), ublas::range(0,j+1)),
      Lambdaprods_r(Lambdaprods_, ublas::range(0,j), ublas::range(0,j+1));  
  ublas::matrix<double> BTilde = ublas::prod(Vprods_r, B_r);
  ublas::matrix<double> CTilde = ublas::prod(Lambdaprods_r, C_r);
  BTilde -= ublas::trans(CTilde);

  y_r += y_norm; // add correction to reduced solution

  // update the predicted decrease in the Augmented Lagrangian and its parameter
  ublas::vector<double> res_proj(j, 0.0);
  for (int k = 0; k < j; k++) res_proj(k) = -g_(0)*Vprods_(k,0);
  res_proj += 0.5*ublas::prod(BTilde, y_r);
  double pred_opt = -ublas::inner_prod(res_proj, y_r);
  res_proj = ublas::prod(CTilde, y_r);
  for (int k = 0; k < j; k++) res_proj(k) -= f_(0)*Lambdaprods_(k,0);
  pred_opt -= ublas::inner_prod(sig_r, res_proj);
  ublas::vector<double> res_red = ublas::prod(C_r, y_r) - f_r;
  double pred_feas = -ublas::inner_prod(res_red, res_red);
  pred_feas += ublas::inner_prod(f_r, f_r); // = f_(0)*f_(0);
  const double rho_mu = 0.01;
  if (pred_feas > kEpsilon) // in case constraints are satisfied
    mu = std::max(mu, -2.0*pred_opt/(pred_feas*(1.0 - rho_mu)));
  pred = pred_opt + 0.5*mu*pred_feas;
}
#endif
// ==============================================================================
} // namespace kona
