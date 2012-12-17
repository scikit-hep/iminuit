#ifndef MN_MnMinimize_H_
#define MN_MnMinimize_H_

#include "Minuit/MnApplication.h"
#include "Minuit/CombinedMinimizer.h"

class FCNBase;

/** API class for minimization using Variable Metric technology ("MIGRAD");
    allows for user interaction: set/change parameters, do minimization,
    change parameters, re-do minimization etc.; 
    also used by MnMinos and MnContours;
 */

class MnMinimize : public MnApplication {

public:

  /// construct from FCNBase + std::vector for parameters and errors
  MnMinimize(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par,err), MnStrategy(stra)), theMinimizer(CombinedMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and covariance
  MnMinimize(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& cov, unsigned int nrow, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov, nrow), MnStrategy(stra)), theMinimizer(CombinedMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and MnUserCovariance
  MnMinimize(const FCNBase& fcn, const std::vector<double>& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(CombinedMinimizer()) {}

  /// construct from FCNBase + MnUserParameters
  MnMinimize(const FCNBase& fcn, const MnUserParameters& par, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par), MnStrategy(stra)), theMinimizer(CombinedMinimizer()) {}

  /// construct from FCNBase + MnUserParameters + MnUserCovariance
  MnMinimize(const FCNBase& fcn, const MnUserParameters& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(CombinedMinimizer()) {}

  /// construct from FCNBase + MnUserParameterState + MnStrategy
  MnMinimize(const FCNBase& fcn, const MnUserParameterState& par, const MnStrategy& str) : MnApplication(fcn, MnUserParameterState(par), str), theMinimizer(CombinedMinimizer()) {}

  MnMinimize(const MnMinimize& migr) : MnApplication(migr.fcnbase(), migr.state(), migr.strategy(), migr.numOfCalls()), theMinimizer(migr.theMinimizer) {}  

  ~MnMinimize() {}

  const ModularFunctionMinimizer& minimizer() const {return theMinimizer;}

private:

  CombinedMinimizer theMinimizer;

private:

  //forbidden assignment of migrad (const FCNBase& = )
  MnMinimize& operator=(const MnMinimize&) {return *this;}
};

#endif //MN_MnMinimize_H_
