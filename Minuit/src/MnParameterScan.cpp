#include "Minuit/MnParameterScan.h"
#include "Minuit/FCNBase.h"

MnParameterScan::MnParameterScan(const FCNBase& fcn, const MnUserParameters& par) : theFCN(fcn), theParameters(par), theAmin(fcn(par.params())) {}

MnParameterScan::MnParameterScan(const FCNBase& fcn, const MnUserParameters& par, double fval) : theFCN(fcn), theParameters(par), theAmin(fval) {}

std::vector<std::pair<double, double> > MnParameterScan::operator()(unsigned int par, unsigned int maxsteps, double low, double high) {

  if(maxsteps > 101) maxsteps = 101;
  std::vector<std::pair<double, double> > result; result.reserve(maxsteps+1);
  std::vector<double> params = theParameters.params();
  result.push_back(std::pair<double, double>(params[par], theAmin));

  if(low > high) return result;
  if(maxsteps < 2) return result;

  if(low == 0. && high == 0.) {
    low = params[par] - 2.*theParameters.error(par);
    high = params[par] + 2.*theParameters.error(par);
  }

  if(low == 0. && high == 0. && theParameters.parameter(par).hasLimits()) {
    if(theParameters.parameter(par).hasLowerLimit())
      low = theParameters.parameter(par).lowerLimit();
    if(theParameters.parameter(par).hasUpperLimit())
      high = theParameters.parameter(par).upperLimit();
  }

  if(theParameters.parameter(par).hasLimits()) {
    if(theParameters.parameter(par).hasLowerLimit())
      low = std::max(low, theParameters.parameter(par).lowerLimit());
    if(theParameters.parameter(par).hasUpperLimit())
      high = std::min(high, theParameters.parameter(par).upperLimit());
  }

  double x0 = low;
  double stp = (high - low)/double(maxsteps - 1);
  for(unsigned int i = 0; i < maxsteps; i++) {
    params[par] = x0 + double(i)*stp;
    double fval = theFCN(params);
    if(fval < theAmin) {
      theParameters.setValue(par, params[par]);
      theAmin = fval;
    }
    result.push_back(std::pair<double, double>(params[par], fval));
  }
 
  return result;
}
