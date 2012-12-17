#include "GaussFcn2.h"
#include "GaussFunction.h"

#include <iostream>
#include <assert.h>

double GaussFcn2::operator()(const std::vector<double>& par) const {

  assert(par.size() == 6);

  GaussFunction gauss1 = GaussFunction(par[0], par[1], par[2]);
  GaussFunction gauss2 = GaussFunction(par[3], par[4], par[5]);

  double chi2 = 0.;
  int nmeas = theMeasurements.size();
  for(int n = 0; n < nmeas; n++) {
    chi2 += ((gauss1(thePositions[n]) + gauss2(thePositions[n]) - theMeasurements[n])*(gauss1(thePositions[n]) + gauss2(thePositions[n]) - theMeasurements[n])/theMVariances[n]);
  }

  return chi2;
}

void GaussFcn2::init() {

  // calculate initial value of chi2

  int nmeas = theMeasurements.size();
  double x = 0.;
  double x2 = 0.;
  double norm = 0.;
  double dx = thePositions[1]-thePositions[0];
  double c = 0.;
  for(int i = 0; i < nmeas; i++) {
    norm += theMeasurements[i];
    x += (theMeasurements[i]*thePositions[i]);
    x2 += (theMeasurements[i]*thePositions[i]*thePositions[i]);
    c += dx*theMeasurements[i];
  }
  double mean = x/norm;
  double rms2 = x2/norm - mean*mean;

//   std::cout<<"FCN initial mean: "<<mean<<std::endl;
//   std::cout<<"FCN initial sigma: "<<sqrt(rms2)<<std::endl;

  std::vector<double> par; 
  par.push_back(mean); par.push_back(sqrt(rms2)); par.push_back(c);
  par.push_back(mean); par.push_back(sqrt(rms2)); par.push_back(c);

  theMin = (*this)(par);
//   std::cout<<"GaussFcnHistoData2 initial chi2: "<<theMin<<std::endl;
         
}

