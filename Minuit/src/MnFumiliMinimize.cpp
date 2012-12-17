#include "Minuit/MnFumiliMinimize.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/FumiliMinimizer.h"


// need to reimplement otherwise base class method is done

FunctionMinimum MnFumiliMinimize::operator()(unsigned int maxfcn, double toler) {

  assert(theState.isValid());
  unsigned int npar = variableParameters();
//   assert(npar > 0);
  if(maxfcn == 0) maxfcn = 200 + 100*npar + 5*npar*npar;
  FunctionMinimum min = minimizer().minimize( fcnbase(), theState, theStrategy, maxfcn, toler);
  theNumCall += min.nfcn();
  theState = min.userState();
  return min;
}
