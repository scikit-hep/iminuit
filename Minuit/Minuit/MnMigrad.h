#ifndef MN_MnMigrad_H_
#define MN_MnMigrad_H_

#include "Minuit/MnApplication.h"
#include "Minuit/VariableMetricMinimizer.h"

class FCNBase;

/** API class for minimization using Variable Metric technology ("MIGRAD");
    allows for user interaction: set/change parameters, do minimization,
    change parameters, re-do minimization etc.; 
    also used by MnMinos and MnContours;
 */

class MnMigrad : public MnApplication {

public:

  /// construct from FCNBase + std::vector for parameters and errors
  MnMigrad(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par,err), MnStrategy(stra)), theMinimizer(VariableMetricMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and covariance
  MnMigrad(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& cov, unsigned int nrow, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov, nrow), MnStrategy(stra)), theMinimizer(VariableMetricMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and MnUserCovariance
  MnMigrad(const FCNBase& fcn, const std::vector<double>& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(VariableMetricMinimizer()) {}

  /// construct from FCNBase + MnUserParameters
  MnMigrad(const FCNBase& fcn, const MnUserParameters& par, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par), MnStrategy(stra)), theMinimizer(VariableMetricMinimizer()) {}

  /// construct from FCNBase + MnUserParameters + MnUserCovariance
  MnMigrad(const FCNBase& fcn, const MnUserParameters& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(VariableMetricMinimizer()) {}

  /// construct from FCNBase + MnUserParameterState + MnStrategy
  MnMigrad(const FCNBase& fcn, const MnUserParameterState& par, const MnStrategy& str) : MnApplication(fcn, MnUserParameterState(par), str), theMinimizer(VariableMetricMinimizer()) {}

  MnMigrad(const MnMigrad& migr) : MnApplication(migr.fcnbase(), migr.state(), migr.strategy(), migr.numOfCalls()), theMinimizer(migr.theMinimizer) {}  

  ~MnMigrad() {}

  const ModularFunctionMinimizer& minimizer() const {return theMinimizer;}

private:

  VariableMetricMinimizer theMinimizer;

private:

  //forbidden assignment of migrad (const FCNBase& = )
  MnMigrad& operator=(const MnMigrad&) {return *this;}
};

#endif //MN_MnMigrad_H_
