#ifndef MN_SimplexParameters_H_
#define MN_SimplexParameters_H_

#include "Minuit/MnMatrix.h"

#include <vector>
#include <utility>

class SimplexParameters {

public:

  SimplexParameters(const std::vector<std::pair<double, MnAlgebraicVector> >& simpl, unsigned int jh, unsigned int jl) : theSimplexParameters(simpl), theJHigh(jh), theJLow(jl) {}

  ~SimplexParameters() {}

  void update(double, const MnAlgebraicVector&); 
  
  const std::vector<std::pair<double, MnAlgebraicVector> >& simplex() const {
    return theSimplexParameters;
  }

  const std::pair<double, MnAlgebraicVector>& operator()(unsigned int i) const {
    assert(i < theSimplexParameters.size());
    return theSimplexParameters[i];
  }
  
  unsigned int jh() const {return theJHigh;}
  unsigned int jl() const {return theJLow;}
  double edm() const {return theSimplexParameters[jh()].first - theSimplexParameters[jl()].first;}
  MnAlgebraicVector dirin() const;

private:

  std::vector<std::pair<double, MnAlgebraicVector> > theSimplexParameters;
  unsigned int theJHigh;
  unsigned int theJLow;
};

#endif //MN_SimplexParameters_H_
