#include "GaussFcn2.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnMinos.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnPrint.h"

#include <iostream>
#include <fstream>


#ifdef USE_SEALBASE
#include "SealBase/Filename.h"
#include "SealBase/ShellEnvironment.h"
#endif



int main() {
  std::vector<double> positions;
  std::vector<double> measurements;
  std::vector<double> var;
  double nmeas = 0;

#ifdef USE_SEALBASE
  seal::Filename   inputFile (seal::Filename ("$SEAL/src/MathLibs/Minuit/tests/MnSim/paul3.txt").substitute (seal::ShellEnvironment ()));
  std::ifstream in(inputFile.name() );
#else
  std::ifstream in("paul3.txt");
#endif
  if (!in) {
    std::cerr << "Error opening input data file" << std::endl;
    return 1; 
  }


  // read input data
  { 
    double x = 0., y = 0., width = 0., err = 0., un1 = 0., un2 = 0.;
    while(in>>x>>y>>width>>err>>un1>>un2) {
      if(err < 1.e-8) continue;
      positions.push_back(x);
      measurements.push_back(y);
      var.push_back(err*err);
      nmeas += y;
      //     xout<<x<<", ";
      //     yout<<y<<", ";
      //     eout<<err*err<<", ";
    }
    std::cout<<"size= "<<var.size()<<std::endl;
    std::cout<<"nmeas: "<<nmeas<<std::endl;
  }

  // create FCN function  
  GaussFcn2 theFCN(measurements, positions, var);

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
  std::vector<double> init_val(6);

  init_val[0] = mean;
  init_val[1] = sqrt(rms2);
  init_val[2] = area; 
  init_val[3] = mean;
  init_val[4] = sqrt(rms2);
  init_val[5] = area; 

// ('Norm', 'Mean', 'Sigma')
// (3286.919999999996, 8676.4053709004238, 3123.5358310131301)

// (1343.311786775236, 10344.596646633145, 3457.8037717416009)
// >>> gauss.parameters()
// (1802.4364028493396, 7090.3704658021443, 1162.144685781906)

  /*
  init_val[0] = 10000.;
  init_val[1] = 3000;
  init_val[2] = 1300; 
  init_val[3] = 7000;
  init_val[4] = 1000;
  init_val[5] = 1800; 
  */
  std::cout<<"initial fval: "<<theFCN(init_val)<<std::endl;

  MnUserParameters upar;
  upar.add("mean1", mean, 10.);
  upar.add("sig1", sqrt(rms2), 10.);
  upar.add("area1", area, 10.);
  upar.add("mean2", mean, 10.);
  upar.add("sig2", sqrt(rms2), 10.);
  upar.add("area2", area, 10.);

  MnMigrad migrad(theFCN, upar);
  std::cout<<"start migrad "<<std::endl;
  FunctionMinimum min = migrad();
  if(!min.isValid()) {
    //try with higher strategy
    std::cout<<"FM is invalid, try with strategy = 2."<<std::endl;
    MnMigrad migrad2(theFCN, upar, 2);
    min = migrad2();
  }
  std::cout<<"minimum: "<<min<<std::endl;
  return 0;
}
