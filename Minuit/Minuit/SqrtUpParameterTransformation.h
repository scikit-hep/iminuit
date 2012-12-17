// Project   : LCG
// Package   : Minuit
// Author    : Lorenzo.MONETA@cern.ch 
// Created by: moneta  at Thu Apr  8 10:26:22 2004

#ifndef MINUIT_SQRTUPPARAMETERTRANSFORMATION_H
#define MINUIT_SQRTUPPARAMETERTRANSFORMATION_H 1

class MnMachinePrecision;

// #include "Minuit/ParameterTransformation.h"

/**
 * Transformation from external to internal parameter based on  sqrt(1 + x**2) 
 * 
 * This transformation applies for the case of single side Upper parameter limits 
 */

class SqrtUpParameterTransformation /* : public ParameterTransformation */ {

public: 
  
  // create with user defined precision
  SqrtUpParameterTransformation() {}
  
  ~SqrtUpParameterTransformation() {}
  
  // transformation from internal to external 
  double int2ext(double value, double upper) const;
  
  // transformation from external to internal
  double ext2int(double value, double upper, const MnMachinePrecision&) const;

  // derivative of transformation from internal to external 
  double dInt2Ext(double value, double upper) const;

private:
  
}; 

#endif 
