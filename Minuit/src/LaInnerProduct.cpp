#include "Minuit/LAVector.h"

double mnddot(unsigned int, const double*, int, const double*, int);

double inner_product(const LAVector& v1, const LAVector& v2) {

  return mnddot(v1.size(), v1.data(), 1, v2.data(), 1);
}
