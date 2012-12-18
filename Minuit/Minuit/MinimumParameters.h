#ifndef MN_MinimumParameters_H_
#define MN_MinimumParameters_H_

#include "Minuit/MnRefCountedPointer.h"
#include "Minuit/BasicMinimumParameters.h"

class MinimumParameters {

public:

  MinimumParameters(unsigned int n) :
   theData(MnRefCountedPointer<BasicMinimumParameters>(new BasicMinimumParameters(n))) {}

  /** takes the parameter vector */
  MinimumParameters(const MnAlgebraicVector& avec, double fval) :
   theData(MnRefCountedPointer<BasicMinimumParameters>(new BasicMinimumParameters(avec, fval)))  {}

  /** takes the parameter vector plus step size x1 - x0 = dirin */
  MinimumParameters(const MnAlgebraicVector& avec, const MnAlgebraicVector& dirin, double fval) : theData(MnRefCountedPointer<BasicMinimumParameters>(new BasicMinimumParameters(avec, dirin, fval)))  {}

  ~MinimumParameters() {}

  MinimumParameters(const MinimumParameters& par) : theData(par.theData) {}

  MinimumParameters& operator=(const MinimumParameters& par) {
    theData = par.theData;
    return *this;
  }

  const MnAlgebraicVector& vec() const {return theData->vec();}
  const MnAlgebraicVector& dirin() const {return theData->dirin();}
  double fval() const {return theData->fval();}
  bool isValid() const {return theData->isValid();}
  bool hasStepSize() const {return theData->hasStepSize();}

private:

  MnRefCountedPointer<BasicMinimumParameters> theData;
};

#endif //MN_MinimumParameters_H_
