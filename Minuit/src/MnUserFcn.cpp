#include "Minuit/MnUserFcn.h"
#include "Minuit/FCNBase.h"
#include "Minuit/MnUserTransformation.h"

double MnUserFcn::operator()(const MnAlgebraicVector& v) const {

  theNumCall++;
  return fcn()( theTransform(v) );
}
