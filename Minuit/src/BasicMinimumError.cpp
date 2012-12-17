#include "Minuit/BasicMinimumError.h"
#include "Minuit/MnPrint.h"


MnAlgebraicSymMatrix BasicMinimumError::hessian() const {
  MnAlgebraicSymMatrix tmp(theMatrix);
  int ifail = invert(tmp);
  if(ifail != 0) {
    std::cout<<"BasicMinimumError inversion fails; return diagonal matrix."<<std::endl;
    MnAlgebraicSymMatrix tmp(theMatrix.nrow());
    for(unsigned int i = 0; i < theMatrix.nrow(); i++) {
      tmp(i,i) = 1./theMatrix(i,i);
    }
    return tmp;
  }
  return tmp;
}
