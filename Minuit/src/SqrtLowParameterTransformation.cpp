// Project   : LCG
// Package   : Minuit
// Author    : Lorenzo.MONETA@cern.ch 
// Created by: moneta  at Thu Apr  8 10:26:22 2004

#include "Minuit/SqrtLowParameterTransformation.h"
#include "Minuit/MnMachinePrecision.h"

/// internal to external transformation
double SqrtLowParameterTransformation::int2ext(double value, double lower) const {

  double val = lower - 1. + sqrt( value*value + 1.);
  return val; 
}

// external to internal transformation
double SqrtLowParameterTransformation::ext2int(double value, double lower, const MnMachinePrecision& prec) const {
  
  double yy = value - lower + 1.; 
  double yy2 = yy*yy; 
  if (yy2 < (1. + prec.eps2()) ) 
    return 8*sqrt(prec.eps2()); 
  else 
    return sqrt( yy2 -1); 
}

// derivative of internal to external transofrmation
double SqrtLowParameterTransformation::dInt2Ext(double value, double) const {

  double val = value/( sqrt( value*value + 1.) );
  return val; 
}
