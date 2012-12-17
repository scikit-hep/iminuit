#ifndef MN_MnParameterScan_H_
#define MN_MnParameterScan_H_

#include "Minuit/MnConfig.h"
#include "Minuit/MnUserParameters.h"

#include <vector>
#include <utility>

class FCNBase;

/** Scans the values of FCN as a function of one parameter and retains the 
    best function and parameter values found.
 */

class MnParameterScan {

public:

  MnParameterScan(const FCNBase&, const MnUserParameters&);

  MnParameterScan(const FCNBase&, const MnUserParameters&, double);

  ~MnParameterScan() {}

// returns pairs of (x,y) points, x=parameter value, y=function value of FCN
  std::vector<std::pair<double, double> > operator()(unsigned int par, unsigned int maxsteps = 41, double low = 0., double high = 0.);

  const MnUserParameters& parameters() const {return theParameters;}
  double fval() const {return theAmin;}

private:

  const FCNBase& theFCN;
  MnUserParameters theParameters;
  double theAmin;
};

#endif //MN_MnParameterScan_H_
