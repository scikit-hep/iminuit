#ifndef MN_MnSimplex_H_
#define MN_MnSimplex_H_

#include "Minuit/MnApplication.h"
#include "Minuit/SimplexMinimizer.h"

class FCNBase;

/** API class for minimization using Variable Metric technology ("MIGRAD");
    allows for user interaction: set/change parameters, do minimization,
    change parameters, re-do minimization etc.; 
    also used by MnMinos and MnContours;
 */

class MnSimplex : public MnApplication {

public:

  /// construct from FCNBase + std::vector for parameters and errors
  MnSimplex(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par,err), MnStrategy(stra)), theMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and covariance
  MnSimplex(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& cov, unsigned int nrow, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov, nrow), MnStrategy(stra)), theMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and MnUserCovariance
  MnSimplex(const FCNBase& fcn, const std::vector<double>& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + MnUserParameters
  MnSimplex(const FCNBase& fcn, const MnUserParameters& par, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par), MnStrategy(stra)), theMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + MnUserParameters + MnUserCovariance
  MnSimplex(const FCNBase& fcn, const MnUserParameters& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + MnUserParameterState + MnStrategy
  MnSimplex(const FCNBase& fcn, const MnUserParameterState& par, const MnStrategy& str) : MnApplication(fcn, MnUserParameterState(par), str), theMinimizer(SimplexMinimizer()) {}

  MnSimplex(const MnSimplex& migr) : MnApplication(migr.fcnbase(), migr.state(), migr.strategy(), migr.numOfCalls()), theMinimizer(migr.theMinimizer) {}  

  ~MnSimplex() {}

  const ModularFunctionMinimizer& minimizer() const {return theMinimizer;}

private:

  SimplexMinimizer theMinimizer;

private:

  //forbidden assignment of migrad (const FCNBase& = )
  MnSimplex& operator=(const MnSimplex&) {return *this;}
};

#endif //MN_MnSimplex_H_
