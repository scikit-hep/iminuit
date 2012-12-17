#include "Minuit/MnCovarianceSqueeze.h"
#include "Minuit/MnUserCovariance.h"
#include "Minuit/MinimumError.h"

#include "Minuit/MnPrint.h"

MnUserCovariance MnCovarianceSqueeze::operator()(const MnUserCovariance& cov, unsigned int n) const {
  
  assert(cov.nrow() > 0);
  assert(n < cov.nrow());
  
  MnAlgebraicSymMatrix hess(cov.nrow());
  for(unsigned int i = 0; i < cov.nrow(); i++) {
    for(unsigned int j = i; j < cov.nrow(); j++) {
      hess(i,j) = cov(i,j);
    }
  }
  
  int ifail = invert(hess);
  
  if(ifail != 0) {
    std::cout<<"MnUserCovariance inversion failed; return diagonal matrix;"<<std::endl;
    MnUserCovariance result(cov.nrow() - 1);
    for(unsigned int i = 0, j =0; i < cov.nrow(); i++) {
      if(i == n) continue;
      result(j,j) = cov(i,i);
      j++;
    }
    return result;
  }
  
  MnAlgebraicSymMatrix squeezed = (*this)(hess, n);

  ifail = invert(squeezed);
  if(ifail != 0) {
    std::cout<<"MnUserCovariance back-inversion failed; return diagonal matrix;"<<std::endl;
    MnUserCovariance result(squeezed.nrow());
    for(unsigned int i = 0; i < squeezed.nrow(); i++) {
      result(i,i) = 1./squeezed(i,i);
    }
    return result;
  }
  
  return MnUserCovariance(std::vector<double>(squeezed.data(), squeezed.data() + squeezed.size()), squeezed.nrow());
}

MinimumError MnCovarianceSqueeze::operator()(const MinimumError& err, unsigned int n) const {
  
  MnAlgebraicSymMatrix hess = err.hessian();
  MnAlgebraicSymMatrix squeezed = (*this)(hess, n);
  int ifail = invert(squeezed);
  if(ifail != 0) {
    std::cout<<"MnCovarianceSqueeze: MinimumError inversion fails; return diagonal matrix."<<std::endl;
    MnAlgebraicSymMatrix tmp(squeezed.nrow());
    for(unsigned int i = 0; i < squeezed.nrow(); i++) {
      tmp(i,i) = 1./squeezed(i,i);
    }
    return MinimumError(tmp, MinimumError::MnInvertFailed());
  }
  
  return MinimumError(squeezed, err.dcovar());
}

MnAlgebraicSymMatrix MnCovarianceSqueeze::operator()(const MnAlgebraicSymMatrix& hess, unsigned int n) const {

  assert(hess.nrow() > 0);
  assert(n < hess.nrow());

  MnAlgebraicSymMatrix hs(hess.nrow() - 1);
  for(unsigned int i = 0, j = 0; i < hess.nrow(); i++) {
    if(i == n) continue;
    for(unsigned int k = i, l = j; k < hess.nrow(); k++) {
      if(k == n) continue;
      hs(j,l) = hess(i,k);
      l++;
    }
    j++;
  }
  
  return hs;
}
