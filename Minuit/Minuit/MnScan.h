#ifndef MN_MnScan_H_
#define MN_MnScan_H_

#include "Minuit/MnApplication.h"
#include "Minuit/ScanMinimizer.h"

class FCNBase;

/** API class for minimization using Variable Metric technology ("MIGRAD");
    allows for user interaction: set/change parameters, do minimization,
    change parameters, re-do minimization etc.; 
    also used by MnMinos and MnContours;
 */

class MnScan : public MnApplication {

public:

  /// construct from FCNBase + std::vector for parameters and errors
  MnScan(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par,err), MnStrategy(stra)), theMinimizer(ScanMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and covariance
  MnScan(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& cov, unsigned int nrow, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov, nrow), MnStrategy(stra)), theMinimizer(ScanMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and MnUserCovariance
  MnScan(const FCNBase& fcn, const std::vector<double>& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(ScanMinimizer()) {}

  /// construct from FCNBase + MnUserParameters
  MnScan(const FCNBase& fcn, const MnUserParameters& par, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par), MnStrategy(stra)), theMinimizer(ScanMinimizer()) {}

  /// construct from FCNBase + MnUserParameters + MnUserCovariance
  MnScan(const FCNBase& fcn, const MnUserParameters& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(ScanMinimizer()) {}

  /// construct from FCNBase + MnUserParameterState + MnStrategy
  MnScan(const FCNBase& fcn, const MnUserParameterState& par, const MnStrategy& str) : MnApplication(fcn, MnUserParameterState(par), str), theMinimizer(ScanMinimizer()) {}

  MnScan(const MnScan& migr) : MnApplication(migr.fcnbase(), migr.state(), migr.strategy(), migr.numOfCalls()), theMinimizer(migr.theMinimizer) {}  

  ~MnScan() {}

  const ModularFunctionMinimizer& minimizer() const {return theMinimizer;}

  std::vector<std::pair<double, double> > scan(unsigned int par, unsigned int maxsteps = 41, double low = 0., double high = 0.);

private:

  ScanMinimizer theMinimizer;

private:

  /// forbidden assignment (const FCNBase& = )
  MnScan& operator=(const MnScan&) {return *this;}
};

#endif //MN_MnScan_H_
