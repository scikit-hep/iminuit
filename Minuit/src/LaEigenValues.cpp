#include "Minuit/LAVector.h"
#include "Minuit/LASymMatrix.h"

int mneigen(double*, unsigned int, unsigned int, unsigned int, double*,double);

LAVector eigenvalues(const LASymMatrix& mat) {

  unsigned int nrow = mat.nrow();
  
  LAVector tmp(nrow*nrow);
  LAVector work(2*nrow);
  
  for(unsigned int i = 0; i < nrow; i++)
    for(unsigned int j = 0; j <= i; j++) {
      tmp(i + j*nrow) = mat(i,j);
      tmp(i*nrow + j) = mat(i,j);
    }

  int info = mneigen(tmp.data(), nrow, nrow, work.size(), work.data(), 1.e-6);

  assert(info == 0);

  LAVector result(nrow);
  for(unsigned int i = 0; i < nrow; i++) result(i) = work(i);

  return result;
}
