#include "equal.hpp"
#include <cmath>

bool nan_equal(double a, double b) {
  return a == b || (std::isnan(a) && std::isnan(b));
}
