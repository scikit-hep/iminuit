#include "Minuit/MnGlobalCorrelationCoeff.h"
#include "Minuit/MnPrint.h"
#include <cmath>

MnGlobalCorrelationCoeff::MnGlobalCorrelationCoeff(const MnAlgebraicSymMatrix& cov) : theGlobalCC(std::vector<double>()), theValid(true) {

  MnAlgebraicSymMatrix inv(cov);
  int ifail = invert(inv);
  if(ifail != 0) {
    std::cout<<"MnGlobalCorrelationCoeff: inversion of matrix fails."<<std::endl;
    theValid = false;
  } else {

    for(unsigned int i = 0; i < cov.nrow(); i++) {
      double denom = inv(i,i)*cov(i,i);
      if(denom < 1. && denom > 0.) theGlobalCC.push_back(0.);
      else theGlobalCC.push_back(std::sqrt(1. - 1./denom));
    }
  }
}
