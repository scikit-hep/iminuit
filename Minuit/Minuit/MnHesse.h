#ifndef MN_MnHesse_H_
#define MN_MnHesse_H_

#include "Minuit/MnConfig.h"
#include "Minuit/MnStrategy.h"

#include <vector>

class FCNBase;
class MnUserParameterState;
class MnUserParameters;
class MnUserCovariance;
class MnUserTransformation;
class MinimumState;
class MnMachinePrecision;
class MnFcn;


/** API class for calculating the numerical covariance matrix 
    (== 2x inverse Hessian == 2x inverse 2nd derivative); can be used by the 
    user or Minuit itself
 */

class MnHesse {

public:

  /// default constructor with default strategy
  MnHesse() : theStrategy(MnStrategy(1)) {}

  /// constructor with user-defined strategy level
  MnHesse(unsigned int stra) : theStrategy(MnStrategy(stra)) {}

  /// conctructor with specific strategy
  MnHesse(const MnStrategy& stra) : theStrategy(stra) {}

  ~MnHesse() {}

  ///
  /// low-level API
  ///
  /// FCN + parameters + errors
  MnUserParameterState operator()(const FCNBase&, const std::vector<double>&, const std::vector<double>&, unsigned int maxcalls=0) const;
  /// FCN + parameters + covariance
  MnUserParameterState operator()(const FCNBase&, const std::vector<double>&, const std::vector<double>&, unsigned int, unsigned int maxcalls=0) const;
  /// FCN + parameters + MnUserCovariance
  MnUserParameterState operator()(const FCNBase&, const std::vector<double>&, const MnUserCovariance&, unsigned int maxcalls=0) const;
  ///
  /// high-level API
  ///
  /// FCN + MnUserParameters
  MnUserParameterState operator()(const FCNBase&, const MnUserParameters&, unsigned int maxcalls=0) const;
  /// FCN + MnUserParameters + MnUserCovariance
  MnUserParameterState operator()(const FCNBase&, const MnUserParameters&, const MnUserCovariance&, unsigned int maxcalls=0) const;
  /// FCN + MnUserParameterState
  MnUserParameterState operator()(const FCNBase&, const MnUserParameterState&, unsigned int maxcalls=0) const;
  ///
  /// internal interface
  ///
  MinimumState operator()(const MnFcn&, const MinimumState&, const MnUserTransformation&, unsigned int maxcalls=0) const;

  /// forward interface of MnStrategy
  unsigned int ncycles() const {return theStrategy.hessianNCycles();}
  double tolerstp() const {return theStrategy.hessianStepTolerance();}
  double tolerg2() const {return theStrategy.hessianG2Tolerance();}

private:

  MnStrategy theStrategy;
};

#endif //MN_MnHesse_H_
