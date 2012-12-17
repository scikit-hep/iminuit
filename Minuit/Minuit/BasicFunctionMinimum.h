#ifndef MN_BasicFunctionMinimum_H_
#define MN_BasicFunctionMinimum_H_

#include "Minuit/MinimumSeed.h"
#include "Minuit/MinimumState.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnUserTransformation.h"

#include "Minuit/StackAllocator.h"

#include <vector>

//extern StackAllocator gStackAllocator;

/** result of the minimization; 
    both internal and external (MnUserParameterState) representation available
    For the parameters at the minimum
 */

class BasicFunctionMinimum {

public:

  class MnReachedCallLimit {};
  class MnAboveMaxEdm {};

public:
  
  BasicFunctionMinimum(const MinimumSeed& seed, double up) : theSeed(seed), theStates(std::vector<MinimumState>(1, MinimumState(seed.parameters(), seed.error(), seed.gradient(), seed.parameters().fval(), seed.nfcn()))), theErrorDef(up), theAboveMaxEdm(false), theReachedCallLimit(false), theUserState(MnUserParameterState()) {}
  
  BasicFunctionMinimum(const MinimumSeed& seed, const std::vector<MinimumState>& states, double up) : theSeed(seed), theStates(states), theErrorDef(up), theAboveMaxEdm(false), theReachedCallLimit(false), theUserState(MnUserParameterState()) {}
  
  BasicFunctionMinimum(const MinimumSeed& seed, const std::vector<MinimumState>& states, double up, MnReachedCallLimit) : theSeed(seed), theStates(states), theErrorDef(up), theAboveMaxEdm(false), theReachedCallLimit(true), theUserState(MnUserParameterState()) {}
  
  BasicFunctionMinimum(const MinimumSeed& seed, const std::vector<MinimumState>& states, double up, MnAboveMaxEdm) : theSeed(seed), theStates(states), theErrorDef(up), theAboveMaxEdm(true), theReachedCallLimit(false), theUserState(MnUserParameterState()) {}

  BasicFunctionMinimum(const BasicFunctionMinimum& min) : theSeed(min.theSeed), theStates(min.theStates), theErrorDef(min.theErrorDef), theAboveMaxEdm(min.theAboveMaxEdm), theReachedCallLimit(min.theReachedCallLimit), theUserState(min.theUserState) {}
  
  BasicFunctionMinimum& operator=(const BasicFunctionMinimum& min) {
    theSeed = min.theSeed;
    theStates = min.theStates;
    theErrorDef = min.theErrorDef;
    theAboveMaxEdm = min.theAboveMaxEdm;
    theReachedCallLimit = min.theReachedCallLimit;
    theUserState = min.theUserState;
    return *this;
  }

  ~BasicFunctionMinimum() {}

// why not
  void add(const MinimumState& state) {
    theStates.push_back(state);
  }

  const MinimumSeed& seed() const {return theSeed;}
  const std::vector<MinimumState>& states() const {return theStates;}

// user representation of state at minimum
  const MnUserParameterState& userState() const {
    if(!theUserState.isValid()) 
      theUserState = MnUserParameterState(state(), up(), seed().trafo());
    return theUserState;
  }
  const MnUserParameters& userParameters() const {
    if(!theUserState.isValid()) 
      theUserState = MnUserParameterState(state(), up(), seed().trafo());
    return theUserState.parameters();
  }
  const MnUserCovariance& userCovariance() const {
    if(!theUserState.isValid()) 
      theUserState = MnUserParameterState(state(), up(), seed().trafo());
    return theUserState.covariance();
  }

  void* operator new(size_t nbytes) {
    return StackAllocatorHolder::get().allocate(nbytes);
  }
  
  void operator delete(void* p, size_t /*nbytes */) {
    StackAllocatorHolder::get().deallocate(p);
  }

// forward interface of last state
  const MinimumState& state() const {return theStates.back();}
  const MinimumParameters& parameters() const {return theStates.back().parameters();}
  const MinimumError& error() const {return theStates.back().error();}
  const FunctionGradient& grad() const {return theStates.back().gradient();}
  double fval() const {return theStates.back().fval();}
  double edm() const {return theStates.back().edm();}
  int nfcn() const {return theStates.back().nfcn();}  
  
  double up() const {return theErrorDef;}
  bool isValid() const {
      return state().isValid() && !isAboveMaxEdm() && !hasReachedCallLimit();
  }
  bool hasValidParameters() const {return state().parameters().isValid();}
  bool hasValidCovariance() const {return state().error().isValid();}
  bool hasAccurateCovar() const {return state().error().isAccurate();}
  bool hasPosDefCovar() const {return state().error().isPosDef();}
  bool hasMadePosDefCovar() const {return state().error().isMadePosDef();}
  bool hesseFailed() const {return state().error().hesseFailed();}
  bool hasCovariance() const {return state().error().isAvailable();}
  bool isAboveMaxEdm() const {return theAboveMaxEdm;}
  bool hasReachedCallLimit() const {return theReachedCallLimit;}

private:

  MinimumSeed theSeed;
  std::vector<MinimumState> theStates;
  double theErrorDef;
  bool theAboveMaxEdm;
  bool theReachedCallLimit;
  mutable MnUserParameterState theUserState;
};

#endif //MN_BasicFunctionMinimum_H_
