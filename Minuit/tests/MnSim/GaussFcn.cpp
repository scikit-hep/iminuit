#include "GaussFcn.h"
#include "GaussFunction.h"

#include <cassert>

double GaussFcn::operator()(const std::vector<double>& par) const {
  
  assert(par.size() == 3);
  GaussFunction gauss(par[0], par[1], par[2]);

  double chi2 = 0.;
  for(unsigned int n = 0; n < theMeasurements.size(); n++) {
    chi2 += ((gauss(thePositions[n]) - theMeasurements[n])*(gauss(thePositions[n]) - theMeasurements[n])/theMVariances[n]);
  }

  return chi2;
}

