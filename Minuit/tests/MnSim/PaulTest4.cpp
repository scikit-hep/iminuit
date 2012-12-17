#include "Minuit/FCNBase.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnMinos.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnPrint.h"
#include "Minuit/SimplexMinimizer.h"

#include <iostream>
#include <fstream>

#ifdef USE_SEALBASE
#include "SealBase/Filename.h"
#include "SealBase/ShellEnvironment.h"
#endif

class PowerLawFunc {

public:

  PowerLawFunc(double p0, double p1) : theP0(p0), theP1(p1) {}

  ~PowerLawFunc() {}

  double operator()(double x) const {
    return p1()*exp(log(x)*p0());
  }

  double p0() const {return theP0;}
  double p1() const {return theP1;}

private:
  double theP0;
  double theP1;
};

class PowerLawChi2FCN : public FCNBase {

public:

  PowerLawChi2FCN(const std::vector<double>& meas,
	      const std::vector<double>& pos,
	      const std::vector<double>& mvar) : theMeasurements(meas),
						 thePositions(pos),
						 theMVariances(mvar) {}

  ~PowerLawChi2FCN() {}

  double operator()(const std::vector<double>& par) const {
    assert(par.size() == 2);
    PowerLawFunc pl(par[0], par[1]);
    double chi2 = 0.;

    for(unsigned int n = 0; n < theMeasurements.size(); n++) {
      chi2 += ((pl(thePositions[n]) - theMeasurements[n])*(pl(thePositions[n]) - theMeasurements[n])/theMVariances[n]);
    }
    
    return chi2;
  }

  double up() const {return 1.;}

private:
  std::vector<double> theMeasurements;
  std::vector<double> thePositions;
  std::vector<double> theMVariances;
};

class PowerLawLogLikeFCN : public FCNBase {

public:

  PowerLawLogLikeFCN(const std::vector<double>& meas, 
		     const std::vector<double>& pos) : 
    theMeasurements(meas), thePositions(pos) {}
  
  ~PowerLawLogLikeFCN() {}
  
  double operator()(const std::vector<double>& par) const {
    assert(par.size() == 2);
    PowerLawFunc pl(par[0], par[1]);
    double logsum = 0.;

    for(unsigned int n = 0; n < theMeasurements.size(); n++) {
      double k = theMeasurements[n];
      double mu = pl(thePositions[n]);
      logsum += (k*log(mu) - mu);
    }
    
    return -logsum;
  }

  double up() const {return 0.5;}

private:
  std::vector<double> theMeasurements;
  std::vector<double> thePositions;
};

int main() {

  std::vector<double> positions;
  std::vector<double> measurements;
  std::vector<double> var;
  {

#ifdef USE_SEALBASE
    seal::Filename   inputFile (seal::Filename ("$SEAL/src/MathLibs/Minuit/tests/MnSim/paul4.txt").substitute (seal::ShellEnvironment ()));
    std::ifstream in(inputFile.name() );
#else
    std::ifstream in("paul4.txt");
#endif
    if (!in) {
      std::cerr << "Error opening input data file" << std::endl;
      return 1; 
  }

    
    double x = 0., y = 0., err = 0.;
    while(in>>x>>y>>err) {
      //       if(err < 1.e-8) continue;
      positions.push_back(x);
      measurements.push_back(y);
      var.push_back(err*err);
    }
    std::cout<<"size= "<<var.size()<<std::endl;
  }
  {
    // create Chi2 FCN function  
    std::cout<<">>> test Chi2"<<std::endl;
    PowerLawChi2FCN theFCN(measurements, positions, var);
    
    MnUserParameters upar;
    upar.add("p0", -2.3, 0.2);
    upar.add("p1", 1100., 10.);
    
    MnMigrad migrad(theFCN, upar);
    std::cout<<"start migrad "<<std::endl;
    FunctionMinimum min = migrad();
    if(!min.isValid()) {
      //try with higher strategy
      std::cout<<"FM is invalid, try with strategy = 2."<<std::endl;
      MnMigrad migrad(theFCN, upar, 2);
      min = migrad();
    }
    std::cout<<"minimum: "<<min<<std::endl;
  }
  {
    std::cout<<">>> test log LikeliHood"<<std::endl;
    // create LogLikelihood FCN function  
    PowerLawLogLikeFCN theFCN(measurements, positions);
    
    MnUserParameters upar;
    upar.add("p0", -2.1, 0.2);
    upar.add("p1", 1000., 10.);
    
    MnMigrad migrad(theFCN, upar);
    std::cout<<"start migrad "<<std::endl;
    FunctionMinimum min = migrad();
    if(!min.isValid()) {
      //try with higher strategy
      std::cout<<"FM is invalid, try with strategy = 2."<<std::endl;
      MnMigrad migrad(theFCN, upar, 2);
      min = migrad();
    }
    std::cout<<"minimum: "<<min<<std::endl;
  }
  {
    std::cout<<">>> test Simplex"<<std::endl;
    PowerLawChi2FCN chi2(measurements, positions, var);
    PowerLawLogLikeFCN mlh(measurements, positions);
    
    MnUserParameters upar;
    std::vector<double> par; par.push_back(-2.3); par.push_back(1100.);
    std::vector<double> err; err.push_back(1.); err.push_back(1.);
    
    SimplexMinimizer simplex;
    
    std::cout<<"start simplex"<<std::endl;
    FunctionMinimum min = simplex.minimize(chi2, par, err);
    std::cout<<"minimum: "<<min<<std::endl;
    FunctionMinimum min2 = simplex.minimize(mlh, par, err);
    std::cout<<"minimum: "<<min2<<std::endl;
  }
  return 0;
}
