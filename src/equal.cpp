#include "equal.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

bool nan_equal(double a, double b) { return std::isnan(a) == std::isnan(b) || a == b; }

namespace std {
bool operator==(const vector<double>& a, const vector<double>& b) {
  return equal(a.begin(), a.end(), b.begin(), b.end());
}
} // namespace std
