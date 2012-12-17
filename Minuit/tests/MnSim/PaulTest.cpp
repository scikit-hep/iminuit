#include "GaussFcn.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnMinos.h"
#include "Minuit/MnPrint.h"

#ifdef USE_SEALBASE
#include "SealBase/Filename.h"
#include "SealBase/ShellEnvironment.h"
#endif


#include <iostream>
#include <fstream>

int main() {

  std::vector<double> positions;
  std::vector<double> measurements;
  std::vector<double> var;
  int nmeas = 0;

#ifdef USE_SEALBASE
  seal::Filename   inputFile (seal::Filename ("$SEAL/src/MathLibs/Minuit/tests/MnSim/paul.txt").substitute (seal::ShellEnvironment ()));
  std::ifstream in(inputFile.name() );
#else
  std::ifstream in("paul.txt");
#endif
  if (!in) {
    std::cerr << "Error opening input data file" << std::endl;
    return 1; 
  }

  // read input data
  { 
    double x = 0., weight = 0., width = 0., err = 0.;
    while(in>>x>>weight>>width>>err) {
      positions.push_back(x);
      double ni = weight*width;
      measurements.push_back(ni);
      var.push_back(ni);
      nmeas += int(ni);
    }
    std::cout<<"size= "<<var.size()<<std::endl;
    assert(var.size() > 0); 
    std::cout<<"nmeas: "<<nmeas<<std::endl;
  }

  // create FCN function  
  GaussFcn theFCN(measurements, positions, var);

  std::vector<double> meas = theFCN.measurements();
  std::vector<double> pos = theFCN.positions();

  // create initial starting values for parameters
  double x = 0.;
  double x2 = 0.;
  double norm = 0.;
  double dx = pos[1]-pos[0];
  double area = 0.;
  for(unsigned int i = 0; i < meas.size(); i++) {
    norm += meas[i];
    x += (meas[i]*pos[i]);
    x2 += (meas[i]*pos[i]*pos[i]);
    area += dx*meas[i];
  }
  double mean = x/norm;
  double rms2 = x2/norm - mean*mean;

  std::cout<<"initial mean: "<<mean<<std::endl;
  std::cout<<"initial sigma: "<<sqrt(rms2)<<std::endl;
  std::cout<<"initial area: "<<area<<std::endl;

  MnUserParameters upar;
  upar.add("mean", mean, 0.1);
  upar.add("sigma", sqrt(rms2), 0.1);
  upar.add("area", area, 0.1);

  MnMigrad migrad(theFCN, upar);
  std::cout<<"start migrad "<<std::endl;
  FunctionMinimum min = migrad();
  std::cout<<"minimum: "<<min<<std::endl;

  std::cout<<"start minos"<<std::endl;
  MnMinos minos(theFCN, min);
  std::pair<double,double> e0 = minos(0);
  std::pair<double,double> e1 = minos(1);
  std::pair<double,double> e2 = minos(2);
  
  std::cout<<"par0: "<<min.userState().value("mean")<<" "<<e0.first<<" "<<e0.second<<std::endl;
  std::cout<<"par1: "<<min.userState().value("sigma")<<" "<<e1.first<<" "<<e1.second<<std::endl;
  std::cout<<"par2: "<<min.userState().value("area")<<" "<<e2.first<<" "<<e2.second<<std::endl;

  return 0;
}
