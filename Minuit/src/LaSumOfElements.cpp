#include "Minuit/LAVector.h"
#include "Minuit/LASymMatrix.h"

double mndasum(unsigned int, const double*, int); 

double sum_of_elements(const LAVector& v) {
  
  return mndasum(v.size(), v.data(), 1);
}

double sum_of_elements(const LASymMatrix& m) {
  
  return mndasum(m.size(), m.data(), 1);
}
