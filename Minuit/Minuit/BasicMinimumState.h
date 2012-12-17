#ifndef MN_BasicMinimumState_H_
#define MN_BasicMinimumState_H_

#include "Minuit/MinimumParameters.h"
#include "Minuit/MinimumError.h"
#include "Minuit/FunctionGradient.h"

#include "Minuit/StackAllocator.h"

//extern StackAllocator gStackAllocator;

class BasicMinimumState {

public:

  BasicMinimumState(unsigned int n) : 
    theParameters(MinimumParameters(n)), theError(MinimumError(n)), 
    theGradient(FunctionGradient(n)), theEDM(0.), theNFcn(0) {}
  BasicMinimumState(const MinimumParameters& states, const MinimumError& err, 
	       const FunctionGradient& grad, double edm, int nfcn) : 
    theParameters(states), theError(err), theGradient(grad), theEDM(edm), theNFcn(nfcn) {}
  
  BasicMinimumState(const MinimumParameters& states, double edm, int nfcn) : theParameters(states), theError(MinimumError(states.vec().size())), theGradient(FunctionGradient(states.vec().size())), theEDM(edm), theNFcn(nfcn) {}
  
  ~BasicMinimumState() {}

  BasicMinimumState(const BasicMinimumState& state) : 
    theParameters(state.theParameters), theError(state.theError), theGradient(state.theGradient), theEDM(state.theEDM), theNFcn(state.theNFcn) {}
  
  BasicMinimumState& operator=(const BasicMinimumState& state) {
    theParameters = state.theParameters; 
    theError = state.theError;
    theGradient = state.theGradient;
    theEDM = state.theEDM;
    theNFcn = state.theNFcn;
    return *this;
  }

  void* operator new(size_t nbytes) {
    return StackAllocatorHolder::get().allocate(nbytes);
  }
  
  void operator delete(void* p, size_t /*nbytes */) {
    StackAllocatorHolder::get().deallocate(p);
  }

  const MinimumParameters& parameters() const {return theParameters;}
  const MnAlgebraicVector& vec() const {return theParameters.vec();}
  int size() const {return theParameters.vec().size();}

  const MinimumError& error() const {return theError;}
  const FunctionGradient& gradient() const {return theGradient;}
  double fval() const {return theParameters.fval();}
  double edm() const {return theEDM;}
  int nfcn() const {return theNFcn;}

  bool isValid() const {    
    if(hasParameters() && hasCovariance()) 
      return parameters().isValid() && error().isValid();
    else if(hasParameters()) return parameters().isValid();
    else return false;
  }  
  bool hasParameters() const {return theParameters.isValid();}
  bool hasCovariance() const {return theError.isAvailable();}
  
private:
  
  MinimumParameters theParameters;
  MinimumError theError;
  FunctionGradient theGradient;
  double theEDM;
  int theNFcn;
};

#endif //MN_BasicMinimumState_H_
