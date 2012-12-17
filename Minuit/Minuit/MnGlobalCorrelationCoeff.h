#ifndef MN_MnGlobalCorrelationCoeff_H_
#define MN_MnGlobalCorrelationCoeff_H_

#include "Minuit/MnConfig.h"
#include "Minuit/MnMatrix.h"

#include <vector>

class MnGlobalCorrelationCoeff {

public:

  MnGlobalCorrelationCoeff() : 
    theGlobalCC(std::vector<double>()), theValid(false) {}

  MnGlobalCorrelationCoeff(const MnAlgebraicSymMatrix&);

  ~MnGlobalCorrelationCoeff() {}

  const std::vector<double>& globalCC() const {return theGlobalCC;}

  bool isValid() const {return theValid;}

private:

  std::vector<double> theGlobalCC;
  bool theValid;
};

#endif //MN_MnGlobalCorrelationCoeff_H_
