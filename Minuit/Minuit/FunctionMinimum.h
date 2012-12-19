#ifndef MN_FunctionMinimum_H_
#define MN_FunctionMinimum_H_

#include "Minuit/BasicFunctionMinimum.h"

/** result of the minimization; 
    both internal and external (MnUserParameterState) representation available
    for the parameters at the minimum
 */

class FunctionMinimum {

public:

  class MnReachedCallLimit {};
  class MnAboveMaxEdm {};

public:
  
  FunctionMinimum(const MinimumSeed& seed, double up) : theData(MnRefCountedPointer<BasicFunctionMinimum>(new BasicFunctionMinimum(seed, up))) {}
  
  FunctionMinimum(const MinimumSeed& seed, const std::vector<MinimumState>& states, double up) : theData(MnRefCountedPointer<BasicFunctionMinimum>(new BasicFunctionMinimum(seed, states, up))) {}
  
  FunctionMinimum(const MinimumSeed& seed, const std::vector<MinimumState>& states, double up, MnReachedCallLimit) : theData(MnRefCountedPointer<BasicFunctionMinimum>(new BasicFunctionMinimum(seed, states, up, BasicFunctionMinimum::MnReachedCallLimit()))) {}
  
  FunctionMinimum(const MinimumSeed& seed, const std::vector<MinimumState>& states, double up, MnAboveMaxEdm) : theData(MnRefCountedPointer<BasicFunctionMinimum>(new BasicFunctionMinimum(seed, states, up, BasicFunctionMinimum::MnAboveMaxEdm()))) {}

  FunctionMinimum(const FunctionMinimum& min) : theData(min.theData) {}
  
  FunctionMinimum& operator=(const FunctionMinimum& min) {
    theData = min.theData;
    return *this;
  }
  
  ~FunctionMinimum() {}
  
  // why not
  void add(const MinimumState& state) {theData->add(state);}

  const MinimumSeed& seed() const {return theData->seed();}
  const std::vector<MinimumState>& states() const {return theData->states();}

// user representation of state at minimum
  const MnUserParameterState& userState() const {
    return theData->userState();
  }
  const MnUserParameters& userParameters() const {
    return theData->userParameters();
  }
  const MnUserCovariance& userCovariance() const {
    return theData->userCovariance();
  }

// forward interface of last state
  const MinimumState& state() const {return theData->state();}
  const MinimumParameters& parameters() const {return theData->parameters();}
  const MinimumError& error() const {return theData->error();}
  const FunctionGradient& grad() const {return theData->grad();}
  double fval() const {return theData->fval();}
  double edm() const {return theData->edm();}
  int nfcn() const {return theData->nfcn();}

  double up() const {return theData->up();}
  bool isValid() const {return theData->isValid();}
  bool isValid_ignoreMaxCall() const{return theData->isValid_ignoreMaxCall();}

  bool hasValidParameters() const {return theData->hasValidParameters();}
  bool hasValidCovariance() const {return theData->hasValidCovariance();}
  bool hasAccurateCovar() const {return theData->hasAccurateCovar();}
  bool hasPosDefCovar() const {return theData->hasPosDefCovar();}
  bool hasMadePosDefCovar() const {return theData->hasMadePosDefCovar();}
  bool hesseFailed() const {return theData->hesseFailed();}
  bool hasCovariance() const {return theData->hasCovariance();}
  bool isAboveMaxEdm() const {return theData->isAboveMaxEdm();}
  bool hasReachedCallLimit() const {return theData->hasReachedCallLimit();}
  void print(bool progress=false) const{return theData->print(progress);}
private:

  MnRefCountedPointer<BasicFunctionMinimum> theData;
};

#endif //MN_FunctionMinimum_H_
