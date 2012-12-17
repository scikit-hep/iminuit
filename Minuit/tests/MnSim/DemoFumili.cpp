#include "GaussDataGen.h"
#include "GaussianModelFunction.h"
#include "Minuit/MnFumiliMinimize.h"
#include "Minuit/FumiliStandardChi2FCN.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnPrint.h"
#include "Minuit/MnMigrad.h"

#include <iostream>

int main() {

  // generate the data (100 data points)
  GaussDataGen gdg(100);

  std::vector<double> pos = gdg.positions();
  std::vector<double> meas = gdg.measurements();
  std::vector<double> var = gdg.variances();

  

  // estimate initial starting values for parameters
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
  double rms = rms2 > 0. ? sqrt(rms2) : 1.;


  // create parameters
  MnUserParameters upar;
  upar.add("mean", mean, 0.1);
  upar.add("sigma", rms, 0.1);
  upar.add("area", area, 0.1);



  // create FCN function for Fumili using model function 
  GaussianModelFunction modelFunction;
  FumiliStandardChi2FCN theFCN(modelFunction, meas, pos, var);
  
  { 

    std::cout << "Minimize using FUMILI : \n" << std::endl; 
    MnFumiliMinimize fumili(theFCN, upar); 

    
    // minimize
    FunctionMinimum min = fumili();

    // output
    std::cout<<"minimum: "<<min<<std::endl;
  }

  {

    std::cout << "Minimize using MIGRAD : \n" << std::endl; 
    MnMigrad migrad(theFCN, upar);

    // minimize
    FunctionMinimum min = migrad();

    // output
    std::cout<<"minimum: "<<min<<std::endl;
  }


  return 0;
}
