#ifndef MN_BasicMinimumError_H_
#define MN_BasicMinimumError_H_

#include "Minuit/MnConfig.h"
#include "Minuit/MnMatrix.h"
#include "Minuit/LaSum.h"
#include "Minuit/StackAllocator.h"
//extern StackAllocator gStackAllocator;

class BasicMinimumError {

public:
 
  class MnNotPosDef {};
  class MnMadePosDef {};
  class MnHesseFailed {};
  class MnInvertFailed {};

public:
  
  BasicMinimumError(unsigned int n) : 
    theMatrix(MnAlgebraicSymMatrix(n)), theDCovar(1.), theValid(false), thePosDef(false), theMadePosDef(false), theHesseFailed(false), theInvertFailed(false), theAvailable(false) {}
 
  BasicMinimumError(const MnAlgebraicSymMatrix& mat, double dcov) : 
    theMatrix(mat), theDCovar(dcov), theValid(true), thePosDef(true), theMadePosDef(false), theHesseFailed(false), theInvertFailed(false), theAvailable(true) {}
  
  BasicMinimumError(const MnAlgebraicSymMatrix& mat, MnHesseFailed) : 
    theMatrix(mat), theDCovar(1.), theValid(false), thePosDef(false), theMadePosDef(false), theHesseFailed(true), theInvertFailed(false), theAvailable(true) {}

  BasicMinimumError(const MnAlgebraicSymMatrix& mat, MnMadePosDef) : 
    theMatrix(mat), theDCovar(1.), theValid(false), thePosDef(false), theMadePosDef(true), theHesseFailed(false), theInvertFailed(false), theAvailable(true) {}

  BasicMinimumError(const MnAlgebraicSymMatrix& mat, MnInvertFailed) : 
    theMatrix(mat), theDCovar(1.), theValid(false), thePosDef(true), theMadePosDef(false), theHesseFailed(false), theInvertFailed(true), theAvailable(true) {}

  BasicMinimumError(const MnAlgebraicSymMatrix& mat, MnNotPosDef) : 
    theMatrix(mat), theDCovar(1.), theValid(false), thePosDef(false), theMadePosDef(false), theHesseFailed(false), theInvertFailed(false), theAvailable(true) {}

  ~BasicMinimumError() {}

  BasicMinimumError(const BasicMinimumError& e) : theMatrix(e.theMatrix), theDCovar(e.theDCovar), theValid(e.theValid), thePosDef(e.thePosDef), theMadePosDef(e.theMadePosDef), theHesseFailed(e.theHesseFailed), theInvertFailed(e.theInvertFailed), theAvailable(e.theAvailable) {}

  BasicMinimumError& operator=(const BasicMinimumError& err) {
    theMatrix = err.theMatrix;
    theDCovar = err.theDCovar;
    theValid = err.theValid;
    thePosDef = err.thePosDef;
    theMadePosDef = err.theMadePosDef;
    theHesseFailed = err.theHesseFailed;
    theInvertFailed = err.theInvertFailed;
    theAvailable = err.theAvailable;
    return *this;
  }

  void* operator new(size_t nbytes) {
    return StackAllocatorHolder::get().allocate(nbytes);
  }
  
  void operator delete(void* p, size_t /*nbytes */) {
    StackAllocatorHolder::get().deallocate(p);
  }

  MnAlgebraicSymMatrix matrix() const {return 2.*theMatrix;}

  const MnAlgebraicSymMatrix& invHessian() const {return theMatrix;}

  MnAlgebraicSymMatrix hessian() const;

  double dcovar() const {return theDCovar;}
  bool isAccurate() const {return theDCovar < 0.1;}
  bool isValid() const {return theValid;}
  bool isPosDef() const {return thePosDef;}
  bool isMadePosDef() const {return theMadePosDef;}
  bool hesseFailed() const {return theHesseFailed;}
  bool invertFailed() const {return theInvertFailed;}
  bool isAvailable() const {return theAvailable;}

private:

  MnAlgebraicSymMatrix theMatrix;
  double theDCovar;
  bool theValid;
  bool thePosDef;
  bool theMadePosDef;
  bool theHesseFailed;
  bool theInvertFailed;
  bool theAvailable;
};

#endif //MN_BasicMinimumError_H_
