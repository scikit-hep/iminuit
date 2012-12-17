#ifndef MN_MinimumError_H_
#define MN_MinimumError_H_

#include "Minuit/MnRefCountedPointer.h"
#include "Minuit/BasicMinimumError.h"
 
/** MinimumError keeps the inv. 2nd derivative (inv. Hessian) used for 
    calculating the parameter step size (-V*g) and for the covariance update
    (ErrorUpdator). The covariance matrix is equal to twice the inv. Hessian.
 */

class MinimumError {

public:

  class MnNotPosDef {};
  class MnMadePosDef {};
  class MnHesseFailed {};
  class MnInvertFailed {};

public:
  
  MinimumError(unsigned int n) : theData(MnRefCountedPointer<BasicMinimumError>(new BasicMinimumError(n))) {}
 
  MinimumError(const MnAlgebraicSymMatrix& mat, double dcov) : theData(MnRefCountedPointer<BasicMinimumError>(new BasicMinimumError(mat, dcov))) {}
  
  MinimumError(const MnAlgebraicSymMatrix& mat, MnHesseFailed) : theData(MnRefCountedPointer<BasicMinimumError>(new BasicMinimumError(mat, BasicMinimumError::MnHesseFailed()))) {}

  MinimumError(const MnAlgebraicSymMatrix& mat, MnMadePosDef) : theData(MnRefCountedPointer<BasicMinimumError>(new BasicMinimumError(mat, BasicMinimumError::MnMadePosDef()))) {}

  MinimumError(const MnAlgebraicSymMatrix& mat, MnInvertFailed) : theData(MnRefCountedPointer<BasicMinimumError>(new BasicMinimumError(mat, BasicMinimumError::MnInvertFailed()))) {}

  MinimumError(const MnAlgebraicSymMatrix& mat, MnNotPosDef) : theData(MnRefCountedPointer<BasicMinimumError>(new BasicMinimumError(mat, BasicMinimumError::MnNotPosDef()))) {}

  ~MinimumError() {}

  MinimumError(const MinimumError& e) : theData(e.theData) {}

  MinimumError& operator=(const MinimumError& err) {
    theData = err.theData;
    return *this;
  }

  MnAlgebraicSymMatrix matrix() const {return theData->matrix();}

  const MnAlgebraicSymMatrix& invHessian() const {return theData->invHessian();}

  MnAlgebraicSymMatrix hessian() const {return theData->hessian();}

  double dcovar() const {return theData->dcovar();}
  bool isAccurate() const {return theData->isAccurate();}
  bool isValid() const {return theData->isValid();}
  bool isPosDef() const {return theData->isPosDef();}
  bool isMadePosDef() const {return theData->isMadePosDef();}
  bool hesseFailed() const {return theData->hesseFailed();}
  bool invertFailed() const {return theData->invertFailed();}
  bool isAvailable() const {return theData->isAvailable();}

private:

  MnRefCountedPointer<BasicMinimumError> theData;
};

#endif //MN_MinimumError_H_
