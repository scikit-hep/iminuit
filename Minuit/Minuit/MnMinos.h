#ifndef MN_MnMinos_H_
#define MN_MnMinos_H_

#include "Minuit/MnStrategy.h"

#include <utility>

class FCNBase;
class FunctionMinimum;
class MinosError;
class MnCross;


/** API class for Minos error analysis (asymmetric errors);
    minimization has to be done before and minimum must be valid;
    possibility to ask only for one side of the Minos error;
 */

class MnMinos {

public:

  /// construct from FCN + minimum
  MnMinos(const FCNBase& fcn, const FunctionMinimum& min) : 
    theFCN(fcn), theMinimum(min), theStrategy(MnStrategy(1)) {} 

  /// construct from FCN + minimum + strategy
  MnMinos(const FCNBase& fcn, const FunctionMinimum& min, unsigned int stra) : 
    theFCN(fcn), theMinimum(min), theStrategy(MnStrategy(stra)) {} 

  /// construct from FCN + minimum + strategy
  MnMinos(const FCNBase& fcn, const FunctionMinimum& min, const MnStrategy& stra) : theFCN(fcn), theMinimum(min), theStrategy(stra) {} 

  ~MnMinos() {}
  
  /// returns the negative (pair.first) and the positive (pair.second) 
  /// minos error of the parameter
  std::pair<double,double> operator()(unsigned int, unsigned int maxcalls = 0) const;

  /// calculate one side (negative or positive error) of the parameter
  double lower(unsigned int, unsigned int maxcalls = 0) const;
  double upper(unsigned int, unsigned int maxcalls = 0) const;

  MnCross loval(unsigned int, unsigned int maxcalls = 0) const;
  MnCross upval(unsigned int, unsigned int maxcalls = 0) const;

  /// ask for MinosError (lower + upper)
  /// can be printed via std::cout  
  MinosError minos(unsigned int, unsigned int maxcalls = 0) const;
  
private:
  
  const FCNBase& theFCN;
  const FunctionMinimum& theMinimum;
  MnStrategy theStrategy;
};

#endif //MN_MnMinos_H_
