#include "Minuit/LaInverse.h"
#include "Minuit/LASymMatrix.h"

int mnvert(LASymMatrix& t);

// symmetric matrix (positive definite only)

int invert(LASymMatrix& t) {

  int ifail = 0;

  if(t.size() == 1) {
    double tmp = t.data()[0];
    if(!(tmp > 0.)) ifail = 1;
    else t.data()[0] = 1./tmp;
  } else {
    ifail = mnvert(t);
  }

  return ifail;
}

