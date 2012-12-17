#ifndef MN_BasicMinimumSeed_H_
#define MN_BasicMinimumSeed_H_

#include "Minuit/MinimumState.h"
#include "Minuit/MinimumParameters.h"
#include "Minuit/MinimumError.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MnUserTransformation.h"

#include "Minuit/StackAllocator.h"

//extern StackAllocator gStackAllocator;

class BasicMinimumSeed {

public:
  
  BasicMinimumSeed(const MinimumState& state, const MnUserTransformation& trafo) : theState(state), theTrafo(trafo), theValid(true) {}
  
  ~BasicMinimumSeed() {}

  BasicMinimumSeed(const BasicMinimumSeed& seed) : theState(seed.theState), theTrafo(seed.theTrafo), theValid(seed.theValid) {}
  
  BasicMinimumSeed& operator=(const BasicMinimumSeed& seed) {
    theState = seed.theState;
    theTrafo = seed.theTrafo;
    theValid = seed.theValid;
    return *this;
  }

  void* operator new(size_t nbytes) {
    return StackAllocatorHolder::get().allocate(nbytes);
  }
  
  void operator delete(void* p, size_t /*nbytes*/) {
    StackAllocatorHolder::get().deallocate(p);
  }

  const MinimumState& state() const {return theState;}
  const MinimumParameters& parameters() const {return state().parameters();}
  const MinimumError& error() const {return state().error();};
  const FunctionGradient& gradient() const {return state().gradient();}
  const MnUserTransformation& trafo() const {return theTrafo;}
  const MnMachinePrecision& precision() const {return theTrafo.precision();}
  double fval() const {return state().fval();}
  double edm() const {return state().edm();}
  unsigned int nfcn() const {return state().nfcn();}
  bool isValid() const {return theValid;}

private:

  MinimumState theState;
  MnUserTransformation theTrafo;
  bool theValid;
};

#endif //MN_BasicMinimumSeed_H_
