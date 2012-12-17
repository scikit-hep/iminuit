#ifndef MN_MnFunctionCross_H_
#define MN_MnFunctionCross_H_

#include "Minuit/MnConfig.h"
#include <vector>


class FCNBase;
class MnUserParameterState;
class MnStrategy;
class MnCross;

/**
   MnFunctionCross 
*/

class MnFunctionCross {

public:
  
  MnFunctionCross(const FCNBase& fcn, const MnUserParameterState& state, double fval, const MnStrategy& stra) : theFCN(fcn), theState(state), theFval(fval), theStrategy(stra) {} 
  
  ~MnFunctionCross() {}
  
  MnCross operator()(const std::vector<unsigned int>&, const std::vector<double>&, const std::vector<double>&, double, unsigned int) const;

private:

  const FCNBase& theFCN;
  const MnUserParameterState& theState;
  double theFval;
  const MnStrategy& theStrategy;
};

#endif //MN_MnFunctionCross_H_
