#ifndef MN_FunctionGradient_H_
#define MN_FunctionGradient_H_

#include "Minuit/MnRefCountedPointer.h"
#include "Minuit/BasicFunctionGradient.h"

class FunctionGradient {

private:

public:
  
  explicit FunctionGradient(unsigned int n) : 
   theData(MnRefCountedPointer<BasicFunctionGradient>(new BasicFunctionGradient(n)))  {}
  
  explicit FunctionGradient(const MnAlgebraicVector& grd) : 
   theData(MnRefCountedPointer<BasicFunctionGradient>(new BasicFunctionGradient(grd))) {}

  FunctionGradient(const MnAlgebraicVector& grd, const MnAlgebraicVector& g2,
		   const MnAlgebraicVector& gstep) : 
   theData(MnRefCountedPointer<BasicFunctionGradient>(new BasicFunctionGradient(grd, g2, gstep))) {}
  
  ~FunctionGradient() {}
  
  FunctionGradient(const FunctionGradient& grad) : theData(grad.theData) {}

  FunctionGradient& operator=(const FunctionGradient& grad) {
    theData = grad.theData;
    return *this;
  }

  const MnAlgebraicVector& grad() const {return theData->grad();}
  const MnAlgebraicVector& vec() const {return theData->vec();}
  bool isValid() const {return theData->isValid();}

  bool isAnalytical() const {return theData->isAnalytical();}
  const MnAlgebraicVector& g2() const {return theData->g2();}
  const MnAlgebraicVector& gstep() const {return theData->gstep();}

private:

  MnRefCountedPointer<BasicFunctionGradient> theData;
};

#endif //MN_FunctionGradient_H_
