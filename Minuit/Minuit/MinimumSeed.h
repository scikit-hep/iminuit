#ifndef MN_MinimumSeed_H_
#define MN_MinimumSeed_H_

#include "Minuit/MnRefCountedPointer.h"
#include "Minuit/BasicMinimumSeed.h"

class MinimumState;
class MinimumParameters;
class MinimumError;
class FunctionGradient;
class MnUserTransformation;

/** MinimumSeed contains the starting values for the minimization produced 
    by the SeedGenerator.
 */

class MinimumSeed {

public:
  
  MinimumSeed(const MinimumState& st, const MnUserTransformation& trafo) : theData(MnRefCountedPointer<BasicMinimumSeed>(new BasicMinimumSeed(st, trafo))) {}
  
  ~MinimumSeed() {}

  MinimumSeed(const MinimumSeed& seed) : theData(seed.theData) {}
  
  MinimumSeed& operator=(const MinimumSeed& seed) {
    theData = seed.theData;
    return *this;
  }

  const MinimumState& state() const {return theData->state();}
  const MinimumParameters& parameters() const {return theData->parameters();}
  const MinimumError& error() const {return theData->error();}
  const FunctionGradient& gradient() const {return theData->gradient();}
  const MnUserTransformation& trafo() const {return theData->trafo();}
  const MnMachinePrecision& precision() const {return theData->precision();}
  double fval() const {return theData->fval();}
  double edm() const {return theData->edm();}
  unsigned int nfcn() const {return theData->nfcn();}
  bool isValid() const {return theData->isValid();}

private:

  MnRefCountedPointer<BasicMinimumSeed> theData;
};

#endif //MN_MinimumSeed_H_
