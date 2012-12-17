#include "Minuit/SinParameterTransformation.h"
#include "Minuit/MnMachinePrecision.h"

#include <math.h>

double SinParameterTransformation::int2ext(double value, double upper, double lower) const {
  
  return lower + 0.5*(upper - lower)*(sin(value) + 1.);
}

double SinParameterTransformation::ext2int(double value, double upper, double lower, const MnMachinePrecision& prec) const {

  double piby2 = 2.*atan(1.);
  double distnn = 8.*sqrt(prec.eps2());
  double vlimhi = piby2 - distnn;
  double vlimlo = -piby2 + distnn;
  
  double yy = 2.*(value - lower)/(upper - lower) - 1.;
  double yy2 = yy*yy;
  if(yy2 > (1. - prec.eps2())) {
    if(yy < 0.) {
      // lower limit
//       std::cout<<"SinParameterTransformation warning: is at its lower allowed limit. "<<value<<std::endl;
      return vlimlo;
    } else {
      // upper limit
//       std::cout<<"SinParameterTransformation warning: is at its upper allowed limit."<<std::endl;
      return vlimhi;
    }
    
  } else {
    return asin(yy); 
  }
}

double SinParameterTransformation::dInt2Ext(double value, double upper, double lower) const {

  return 0.5*fabs((upper - lower)*cos(value));
}
