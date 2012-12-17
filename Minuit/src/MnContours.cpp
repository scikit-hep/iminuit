#include "Minuit/MnContours.h"
#include "Minuit/MnMinos.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnFunctionCross.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/FCNBase.h"
#include "Minuit/MnCross.h"
#include "Minuit/MinosError.h"
#include "Minuit/ContoursError.h"

#include "Minuit/MnPrint.h"

std::vector<std::pair<double,double> > MnContours::operator()(unsigned int px, unsigned int py, unsigned int npoints) const {

  ContoursError cont = contour(px, py, npoints);
  return cont();
}

ContoursError MnContours::contour(unsigned int px, unsigned int py, unsigned int npoints) const {

  assert(npoints > 3);
  unsigned int maxcalls = 100*(npoints+5)*(theMinimum.userState().variableParameters()+1);
  unsigned int nfcn = 0;

  std::vector<std::pair<double,double> > result; result.reserve(npoints);
  std::vector<MnUserParameterState> states;
//   double edmmax = 0.5*0.05*theFCN.up()*1.e-3;    
  double toler = 0.05;    
  
  //get first four points
//   std::cout<<"MnContours: get first 4 params."<<std::endl;
  MnMinos minos(theFCN, theMinimum, theStrategy);
  
  double valx = theMinimum.userState().value(px);
  double valy = theMinimum.userState().value(py);

  MinosError mex = minos.minos(px);
  nfcn += mex.nfcn();
  if(!mex.isValid()) {
    std::cout<<"MnContours is unable to find first two points."<<std::endl;
    return ContoursError(px, py, result, mex, mex, nfcn);
  }
  std::pair<double,double> ex = mex();

  MinosError mey = minos.minos(py);
  nfcn += mey.nfcn();
  if(!mey.isValid()) {
    std::cout<<"MnContours is unable to find second two points."<<std::endl;
    return ContoursError(px, py, result, mex, mey, nfcn);
  }
  std::pair<double,double> ey = mey();

  MnMigrad migrad(theFCN, theMinimum.userState(), MnStrategy(std::max(0, int(theStrategy.strategy()-1))));

  migrad.fix(px);
  migrad.setValue(px, valx + ex.second);
  FunctionMinimum exy_up = migrad();
  nfcn += exy_up.nfcn();
  if(!exy_up.isValid()) {
    std::cout<<"MnContours is unable to find upper y value for x parameter "<<px<<"."<<std::endl;
    return ContoursError(px, py, result, mex, mey, nfcn);
  }

  migrad.setValue(px, valx + ex.first);
  FunctionMinimum exy_lo = migrad();
  nfcn += exy_lo.nfcn();
  if(!exy_lo.isValid()) {
    std::cout<<"MnContours is unable to find lower y value for x parameter "<<px<<"."<<std::endl;
    return ContoursError(px, py, result, mex, mey, nfcn);
  }

  
  MnMigrad migrad1(theFCN, theMinimum.userState(), MnStrategy(std::max(0, int(theStrategy.strategy()-1))));
  migrad1.fix(py);
  migrad1.setValue(py, valy + ey.second);
  FunctionMinimum eyx_up = migrad1();
  nfcn += eyx_up.nfcn();
  if(!eyx_up.isValid()) {
    std::cout<<"MnContours is unable to find upper x value for y parameter "<<py<<"."<<std::endl;
    return ContoursError(px, py, result, mex, mey, nfcn);
  }

  migrad1.setValue(py, valy + ey.first);
  FunctionMinimum eyx_lo = migrad1();
  nfcn += eyx_lo.nfcn();
  if(!eyx_lo.isValid()) {
    std::cout<<"MnContours is unable to find lower x value for y parameter "<<py<<"."<<std::endl;
    return ContoursError(px, py, result, mex, mey, nfcn);
  }
  
  double scalx = 1./(ex.second - ex.first);
  double scaly = 1./(ey.second - ey.first);

  result.push_back(std::pair<double,double>(valx + ex.first, exy_lo.userState().value(py)));
  result.push_back(std::pair<double,double>(eyx_lo.userState().value(px), valy + ey.first));
  result.push_back(std::pair<double,double>(valx + ex.second, exy_up.userState().value(py)));
  result.push_back(std::pair<double,double>(eyx_up.userState().value(px), valy + ey.second));

//   std::cout<<"MnContours: first 4 params finished."<<std::endl;

  MnUserParameterState upar = theMinimum.userState();
  upar.fix(px);
  upar.fix(py);

  std::vector<unsigned int> par(2); par[0] = px; par[1] = py;
  MnFunctionCross cross(theFCN, upar, theMinimum.fval(), theStrategy);

  for(unsigned int i = 4; i < npoints; i++) {
    
    std::vector<std::pair<double,double> >::iterator idist1 = result.end()-1;
    std::vector<std::pair<double,double> >::iterator idist2 = result.begin();
    double distx = idist1->first - (idist2)->first;
    double disty = idist1->second - (idist2)->second;
    double bigdis = scalx*scalx*distx*distx + scaly*scaly*disty*disty;
    
    for(std::vector<std::pair<double,double> >::iterator ipair = result.begin(); ipair != result.end()-1; ipair++) {
      double distx = ipair->first - (ipair+1)->first;
      double disty = ipair->second - (ipair+1)->second;
      double dist = scalx*scalx*distx*distx + scaly*scaly*disty*disty;
      if(dist > bigdis) {
	bigdis = dist;
	idist1 = ipair;
	idist2 = ipair+1;
      }
    }
    
    double a1 = 0.5;
    double a2 = 0.5;
    double sca = 1.;

L300:

    if(nfcn > maxcalls) {
      std::cout<<"MnContours: maximum number of function calls exhausted."<<std::endl;
      return ContoursError(px, py, result, mex, mey, nfcn);
    }

    double xmidcr = a1*idist1->first + a2*(idist2)->first;
    double ymidcr = a1*idist1->second + a2*(idist2)->second;
    double xdir = (idist2)->second - idist1->second;
    double ydir = idist1->first - (idist2)->first;
    double scalfac = sca*std::max(fabs(xdir*scalx), fabs(ydir*scaly));
    double xdircr = xdir/scalfac;
    double ydircr = ydir/scalfac;
    std::vector<double> pmid(2); pmid[0] = xmidcr; pmid[1] = ymidcr;
    std::vector<double> pdir(2); pdir[0] = xdircr; pdir[1] = ydircr;

    MnCross opt = cross(par, pmid, pdir, toler, maxcalls);
    nfcn += opt.nfcn();
    if(!opt.isValid()) {
//       if(a1 > 0.5) {
      if(sca < 0.) {
	std::cout<<"MnContours is unable to find point "<<i+1<<" on contour."<<std::endl;
	std::cout<<"MnContours finds only "<<i<<" points."<<std::endl;
	return ContoursError(px, py, result, mex, mey, nfcn);
      }
//       a1 = 0.75;
//       a2 = 0.25;
//       std::cout<<"*****switch direction"<<std::endl;
      sca = -1.;
      goto L300;
    }
    double aopt = opt.value();
    if(idist2 == result.begin())
      result.push_back(std::pair<double,double>(xmidcr+(aopt)*xdircr, ymidcr + (aopt)*ydircr));
    else 
      result.insert(idist2, std::pair<double,double>(xmidcr+(aopt)*xdircr, ymidcr + (aopt)*ydircr));
  }

  return ContoursError(px, py, result, mex, mey, nfcn);
}
