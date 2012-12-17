#ifndef MN_MnContours_H_
#define MN_MnContours_H_


#include "Minuit/MnConfig.h"
#include "Minuit/MnStrategy.h"

#include <vector>
#include <utility>

class FCNBase;
class FunctionMinimum;
class ContoursError;

/**
   API class for Contours error analysis (2-dim errors);
   minimization has to be done before and minimum must be valid;
   possibility to ask only for the points or the points and associated Minos
   errors;
 */

class MnContours {

public:

  /// construct from FCN + minimum
  MnContours(const FCNBase& fcn, const FunctionMinimum& min) : theFCN(fcn), theMinimum(min), theStrategy(MnStrategy(1)) {} 

  /// construct from FCN + minimum + strategy
  MnContours(const FCNBase& fcn, const FunctionMinimum& min, unsigned int stra) : theFCN(fcn), theMinimum(min), theStrategy(MnStrategy(stra)) {} 

  /// construct from FCN + minimum + strategy
  MnContours(const FCNBase& fcn, const FunctionMinimum& min, const MnStrategy& stra) : theFCN(fcn), theMinimum(min), theStrategy(stra) {} 

  ~MnContours() {}

  /// ask for one contour (points only)
  std::vector<std::pair<double,double> > operator()(unsigned int, unsigned int, unsigned int npoints = 20) const;

  /// ask for one contour ContoursError (MinosErrors + points)
  /// can be printed via std::cout
  ContoursError contour(unsigned int, unsigned int, unsigned int npoints = 20) const;

  const MnStrategy& strategy() const {return theStrategy;}

private:

  const FCNBase& theFCN;
  const FunctionMinimum& theMinimum;
  MnStrategy theStrategy;
};

#endif //MN_MnContours_H_
