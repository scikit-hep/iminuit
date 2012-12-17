#ifndef MN_BasicFunctionGradient_H_
#define MN_BasicFunctionGradient_H_

#include "Minuit/MnMatrix.h"

#include "Minuit/StackAllocator.h"

//extern StackAllocator gStackAllocator;

class BasicFunctionGradient {

private:

public:
  
  explicit BasicFunctionGradient(unsigned int n) :
    theGradient(MnAlgebraicVector(n)), theG2ndDerivative(MnAlgebraicVector(n)),
    theGStepSize(MnAlgebraicVector(n)), theValid(false), 
    theAnalytical(false) {}
  
  explicit BasicFunctionGradient(const MnAlgebraicVector& grd) : 
    theGradient(grd), theG2ndDerivative(MnAlgebraicVector(grd.size())),
    theGStepSize(MnAlgebraicVector(grd.size())), theValid(true), 
    theAnalytical(true) {}

  BasicFunctionGradient(const MnAlgebraicVector& grd, const MnAlgebraicVector& g2, const MnAlgebraicVector& gstep) : 
    theGradient(grd), theG2ndDerivative(g2),
    theGStepSize(gstep), theValid(true), theAnalytical(false) {}
  
  ~BasicFunctionGradient() {}
  
  BasicFunctionGradient(const BasicFunctionGradient& grad) : theGradient(grad.theGradient), theG2ndDerivative(grad.theG2ndDerivative), theGStepSize(grad.theGStepSize), theValid(grad.theValid) {}

  BasicFunctionGradient& operator=(const BasicFunctionGradient& grad) {
    theGradient = grad.theGradient;
    theG2ndDerivative = grad.theG2ndDerivative;
    theGStepSize = grad.theGStepSize;
    theValid = grad.theValid;
    return *this;
  }

  void* operator new(size_t nbytes) {
    return StackAllocatorHolder::get().allocate(nbytes);
  }
  
  void operator delete(void* p, size_t /*nbytes */) {
    StackAllocatorHolder::get().deallocate(p);
  }

  const MnAlgebraicVector& grad() const {return theGradient;}
  const MnAlgebraicVector& vec() const {return theGradient;}
  bool isValid() const {return theValid;}

  bool isAnalytical() const {return theAnalytical;}
  const MnAlgebraicVector& g2() const {return theG2ndDerivative;}
  const MnAlgebraicVector& gstep() const {return theGStepSize;}

private:

  MnAlgebraicVector theGradient;
  MnAlgebraicVector theG2ndDerivative;
  MnAlgebraicVector theGStepSize;
  bool theValid;
  bool theAnalytical;
};

#endif //MN_BasicFunctionGradient_H_
