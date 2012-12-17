#ifndef MN_MinosError_H_
#define MN_MinosError_H_

#include "Minuit/MnCross.h"
#include <iostream>

class MinosError {

public:

  MinosError() : theParameter(0), theMinValue(0.), theUpper(MnCross()), theLower(MnCross()) {}
  
  MinosError(unsigned int par, double min, const MnCross& low, const MnCross& up) : theParameter(par), theMinValue(min), theUpper(up), theLower(low) {}

  ~MinosError() {}

  MinosError(const MinosError& err) : theParameter(err.theParameter), theMinValue(err.theMinValue), theUpper(err.theUpper),  theLower(err.theLower) {}

  MinosError& operator()(const MinosError& err) {
    theParameter = err.theParameter;
    theMinValue = err.theMinValue;
    theUpper = err.theUpper;
    theLower = err.theLower;
    return *this;
  }

  std::pair<double,double> operator()() const {
    return std::pair<double,double>(lower(), upper());
  }
  double lower() const {
    return -1.*lowerState().error(parameter())*(1. + theLower.value());
  }
  double upper() const {
    return upperState().error(parameter())*(1. + theUpper.value());
  }
  unsigned int parameter() const {return theParameter;}
  const MnUserParameterState& lowerState() const {return theLower.state();}
  const MnUserParameterState& upperState() const {return theUpper.state();}
  bool isValid() const {return theLower.isValid() && theUpper.isValid();}
  bool lowerValid() const {return theLower.isValid();}
  bool upperValid() const {return theUpper.isValid();}
  bool atLowerLimit() const {return theLower.atLimit();}
  bool atUpperLimit() const {return theUpper.atLimit();}
  bool atLowerMaxFcn() const {return theLower.atMaxFcn();}
  bool atUpperMaxFcn() const {return theUpper.atMaxFcn();}
  bool lowerNewMin() const {return theLower.newMinimum();}
  bool upperNewMin() const {return theUpper.newMinimum();}
  unsigned int nfcn() const {return theUpper.nfcn() + theLower.nfcn();}
  double min() const {return theMinValue;}

private:
  
  unsigned int theParameter;
  double theMinValue;
  MnCross theUpper;
  MnCross theLower;
};

#endif //MN_MinosError_H_
