#ifndef MN_BasicMinimumParameters_H_
#define MN_BasicMinimumParameters_H_

#include "Minuit/MnMatrix.h"

#include "Minuit/StackAllocator.h"

//extern StackAllocator gStackAllocator;

class BasicMinimumParameters {

public:

  BasicMinimumParameters(unsigned int n) : theParameters(MnAlgebraicVector(n)), theStepSize(MnAlgebraicVector(n)), theFVal(0.), theValid(false), theHasStep(false) {}
  
  BasicMinimumParameters(const MnAlgebraicVector& avec, double fval) : 
    theParameters(avec), theStepSize(avec.size()), theFVal(fval), theValid(true), theHasStep(false) {}
  
  BasicMinimumParameters(const MnAlgebraicVector& avec, const MnAlgebraicVector& dirin, double fval) : theParameters(avec), theStepSize(dirin), theFVal(fval), theValid(true), theHasStep(true) {}
  
  ~BasicMinimumParameters() {}

  BasicMinimumParameters(const BasicMinimumParameters& par) : theParameters(par.theParameters), theStepSize(par.theStepSize), theFVal(par.theFVal), theValid(par.theValid), theHasStep(par.theHasStep) {}

  BasicMinimumParameters& operator=(const BasicMinimumParameters& par) {
    theParameters = par.theParameters;
    theStepSize = par.theStepSize;
    theFVal = par.theFVal;
    theValid = par.theValid; 
    theHasStep = par.theHasStep;
    return *this;
  }

  void* operator new(size_t nbytes) {
    return StackAllocatorHolder::get().allocate(nbytes);
  }
  
  void operator delete(void* p, size_t /*nbytes*/) {
    StackAllocatorHolder::get().deallocate(p);
  }

  const MnAlgebraicVector& vec() const {return theParameters;}
  const MnAlgebraicVector& dirin() const {return theStepSize;}
  double fval() const {return theFVal;}
  bool isValid() const {return theValid;}
  bool hasStepSize() const {return theHasStep;}

private:

  MnAlgebraicVector theParameters;
  MnAlgebraicVector theStepSize;
  double theFVal;
  bool theValid;
  bool theHasStep;
};

#endif //MN_BasicMinimumParameters_H_
