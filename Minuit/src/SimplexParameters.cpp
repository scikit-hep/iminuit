#include "Minuit/SimplexParameters.h"

void SimplexParameters::update(double y, const MnAlgebraicVector& p) {

  theSimplexParameters[jh()] = std::pair<double, MnAlgebraicVector>(y, p);
  if(y < theSimplexParameters[jl()].first) theJLow = jh();

  unsigned int jh = 0;
  for(unsigned int i = 1; i < theSimplexParameters.size(); i++) {
    if(theSimplexParameters[i].first > theSimplexParameters[jh].first) jh = i;
  }
  theJHigh = jh;

  return;
} 

MnAlgebraicVector SimplexParameters::dirin() const {

  MnAlgebraicVector dirin(theSimplexParameters.size() - 1);
  for(unsigned int i = 0; i < theSimplexParameters.size() - 1; i++) {
    double pbig = theSimplexParameters[0].second(i), plit = pbig;
    for(unsigned int j = 0; j < theSimplexParameters.size(); j++){
      if(theSimplexParameters[j].second(i) < plit) plit = theSimplexParameters[j].second(i);
      if(theSimplexParameters[j].second(i) > pbig) pbig = theSimplexParameters[j].second(i);
    }
    dirin(i) = pbig - plit;
  } 

  return dirin;
}
