// Project   : LCG
// Package   : Minuit
// Author    : Lorenzo.MONETA@cern.ch 
// Created by: moneta  at Thu Apr  8 10:26:22 2004

#ifndef MINUIT_SQRTLOWPARAMETERTRANSFORMATION_H
#define MINUIT_SQRTLOWPARAMETERTRANSFORMATION_H 1

class MnMachinePrecision;

// #include "Minuit/ParameterTransformation.h"

/**
 * Transformation from external to internal parameter based on  sqrt(1 + x**2) 
 * 
 * This transformation applies for the case of single side Lower parameter limits 
 */

class SqrtLowParameterTransformation /* : public ParameterTransformation */ {

public: 
  
  SqrtLowParameterTransformation() {}

  ~SqrtLowParameterTransformation() {}

  // transformation from internal to external 
  double int2ext(double value, double lower) const;
  
  // transformation from external to internal
  double ext2int(double value, double lower, const MnMachinePrecision&) const;

  // derivative of transformation from internal to external 
  double dInt2Ext(double value, double lower) const;

private:
}; 

#endif 
