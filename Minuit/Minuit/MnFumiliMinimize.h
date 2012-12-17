#ifndef MN_MnFumiliMinimize_H_
#define MN_MnFumiliMinimize_H_

#include "Minuit/MnApplication.h"
#include "Minuit/FumiliMinimizer.h"
#include "Minuit/FumiliFCNBase.h"

// class FumiliFCNBase;
// class FCNBase;

/** 


API class for minimization using Fumili technology;
allows for user interaction: set/change parameters, do minimization,
change parameters, re-do minimization etc.; 
also used by MnMinos and MnContours;

\todo This a first try and not yet guaranteed at all to work

 */

class MnFumiliMinimize : public MnApplication {

public:

  /// construct from FumiliFCNBase + std::vector for parameters and errors
  MnFumiliMinimize(const FumiliFCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par,err), MnStrategy(stra)), theMinimizer(FumiliMinimizer()), theFCN(fcn) {}

  /// construct from FumiliFCNBase + std::vector for parameters and covariance
  MnFumiliMinimize(const FumiliFCNBase& fcn, const std::vector<double>& par, const std::vector<double>& cov, unsigned int nrow, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov, nrow), MnStrategy(stra)), theMinimizer(FumiliMinimizer()), theFCN(fcn) {}

  /// construct from FumiliFCNBase + std::vector for parameters and MnUserCovariance
  MnFumiliMinimize(const FumiliFCNBase& fcn, const std::vector<double>& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(FumiliMinimizer()), theFCN(fcn) {}

  /// construct from FumiliFCNBase + MnUserParameters
  MnFumiliMinimize(const FumiliFCNBase& fcn, const MnUserParameters& par, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par), MnStrategy(stra)), theMinimizer(FumiliMinimizer()), theFCN(fcn) {}

  /// construct from FumiliFCNBase + MnUserParameters + MnUserCovariance
  MnFumiliMinimize(const FumiliFCNBase& fcn, const MnUserParameters& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), theMinimizer(FumiliMinimizer()), theFCN(fcn) {}

  /// construct from FumiliFCNBase + MnUserParameterState + MnStrategy
  MnFumiliMinimize(const FumiliFCNBase& fcn, const MnUserParameterState& par, const MnStrategy& str) : MnApplication(fcn, MnUserParameterState(par), str), theMinimizer(FumiliMinimizer()), theFCN(fcn) {}

  MnFumiliMinimize(const MnFumiliMinimize& migr) : MnApplication(migr.fcnbase(), migr.state(), migr.strategy(), migr.numOfCalls()), theMinimizer(migr.theMinimizer), theFCN(migr.fcnbase()) {}  

  virtual ~MnFumiliMinimize() { }

  const FumiliMinimizer& minimizer() const {return theMinimizer;}

  const FumiliFCNBase & fcnbase() const { return theFCN; }


  /// overwrite minimize to use FumiliFCNBase
  virtual FunctionMinimum operator()(unsigned int = 0, double = 0.1);


private:

  FumiliMinimizer theMinimizer;
  const FumiliFCNBase & theFCN;

private:

  //forbidden assignment of migrad (const FumiliFCNBase& = )
  MnFumiliMinimize& operator=(const MnFumiliMinimize&) {return *this;}
};

#endif //MN_MnFumiliMinimize_H_
