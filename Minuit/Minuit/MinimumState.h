#ifndef MN_MinimumState_H_
#define MN_MinimumState_H_

#include "Minuit/MnRefCountedPointer.h"
#include "Minuit/BasicMinimumState.h"

class MinimumParameters;
class MinimumError;
class FunctionGradient;

/** MinimumState keeps the information (position, gradient, 2nd deriv, etc)
    after one minimization step (usually in MinimumBuilder).
 */

class MinimumState {

public:

  /** invalid state */
  MinimumState(unsigned int n) :
    theData(MnRefCountedPointer<BasicMinimumState>(new BasicMinimumState(n))) {}
  /** state with parameters only (from stepping methods like Simplex, Scan) */
  MinimumState(const MinimumParameters& states, double edm, int nfcn) :
    theData(MnRefCountedPointer<BasicMinimumState>(new BasicMinimumState(states, edm, nfcn))) {}

  /** state with parameters, gradient and covariance (from gradient methods
      such as Migrad) */
  MinimumState(const MinimumParameters& states, const MinimumError& err,
	       const FunctionGradient& grad, double edm, int nfcn) :
    theData(MnRefCountedPointer<BasicMinimumState>(new BasicMinimumState(states, err, grad, edm, nfcn))) {}

  ~MinimumState() {}

  MinimumState(const MinimumState& state) : theData(state.theData) {}

  MinimumState& operator=(const MinimumState& state) {
    theData = state.theData;
    return *this;
  }

  const MinimumParameters& parameters() const {return theData->parameters();}
  const MnAlgebraicVector& vec() const {return theData->vec();}
  int size() const {return theData->size();}

  const MinimumError& error() const {return theData->error();}
  const FunctionGradient& gradient() const {return theData->gradient();}
  double fval() const {return theData->fval();}
  double edm() const {return theData->edm();}
  int nfcn() const {return theData->nfcn();}

  bool isValid() const {return theData->isValid();}

  bool hasParameters() const {return theData->hasParameters();}
  bool hasCovariance() const {return theData->hasCovariance();}

private:

  MnRefCountedPointer<BasicMinimumState> theData;
};

#endif //MN_MinimumState_H_
