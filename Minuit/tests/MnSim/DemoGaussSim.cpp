#include "GaussFcn.h"
#include "GaussDataGen.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnPrint.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnMinos.h"
#include "Minuit/MnContours.h"
#include "Minuit/MnPlot.h"
#include "Minuit/MinosError.h"
#include "Minuit/ContoursError.h"

#include <iostream>

int main() {

  // generate the data (100 data points)
  GaussDataGen gdg(100);

  std::vector<double> pos = gdg.positions();
  std::vector<double> meas = gdg.measurements();
  std::vector<double> var = gdg.variances();
   
  // create FCN function  
  GaussFcn theFCN(meas, pos, var);

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
  double rms = rms2 > 0. ? sqrt(rms2) : 1.;

  {
    // demonstrate minimal required interface for minimization
    // create Minuit parameters without names

    // starting values for parameters
    std::vector<double> init_par; 
    init_par.push_back(mean); 
    init_par.push_back(rms); 
    init_par.push_back(area);

    // starting values for initial uncertainties
    std::vector<double> init_err; 
    init_err.push_back(0.1); 
    init_err.push_back(0.1); 
    init_err.push_back(0.1);
    
    // create minimizer (default constructor)
    VariableMetricMinimizer theMinimizer;
    
    // minimize
    FunctionMinimum min = theMinimizer.minimize(theFCN, init_par, init_err);

    // output
    std::cout<<"minimum: "<<min<<std::endl;
  }

  {
    // demonstrate standard minimization using MIGRAD
    // create Minuit parameters with names
    MnUserParameters upar;
    upar.add("mean", mean, 0.1);
    upar.add("sigma", rms, 0.1);
    upar.add("area", area, 0.1);

    // create MIGRAD minimizer
    MnMigrad migrad(theFCN, upar);

    // minimize
    FunctionMinimum min = migrad();

    // output
    std::cout<<"minimum: "<<min<<std::endl;
  }

  {
    // demonstrate full interaction with parameters over subsequent 
    // minimizations

    // create Minuit parameters with names
    MnUserParameters upar;
    upar.add("mean", mean, 0.1);
    upar.add("sigma", rms, 0.1);
    upar.add("area", area, 0.1);

    // access parameter by name to set limits...
    upar.setLimits("mean", mean-0.01, mean+0.01);

    // ... or access parameter by index
    upar.setLimits(1, rms-0.1, rms+0.1);
    
    // create Migrad minimizer
    MnMigrad migrad(theFCN, upar);

    // fix a parameter...
    migrad.fix("mean");

    // ... and minimize
    FunctionMinimum min = migrad();

    // output
    std::cout<<"minimum: "<<min<<std::endl;

    // release a parameter...
    migrad.release("mean");

    // ... and fix another one
    migrad.fix(1);

    // and minimize again
    FunctionMinimum min1 = migrad();
 
    // output
    std::cout<<"minimum1: "<<min1<<std::endl;

    // release the parameter...
    migrad.release(1);

    // ... and minimize with all three parameters (still with limits!)
    FunctionMinimum min2 = migrad();
    
    // output
    std::cout<<"minimum2: "<<min2<<std::endl;

    // remove all limits on parameters...
    migrad.removeLimits("mean");
    migrad.removeLimits("sigma");

    // ... and minimize again with all three parameters (now without limits!)
    FunctionMinimum min3 = migrad();

    // output
    std::cout<<"minimum3: "<<min3<<std::endl;
  }

  {
    // test single sided limits
    MnUserParameters upar;
    upar.add("mean", mean, 0.1);
    upar.add("sigma", rms-1., 0.1);
    upar.add("area", area, 0.1);

    // test lower limits
    upar.setLowerLimit("mean", mean-0.01);

    // test upper limits
    upar.setUpperLimit("sigma", rms-0.5);

    // create MIGRAD minimizer
    MnMigrad migrad(theFCN, upar);

    // ... and minimize
    FunctionMinimum min = migrad();
    std::cout<<"test lower limit minimim= "<<min<<std::endl;
  }

  {
    // demonstrate MINOS error analysis

    // create Minuit parameters with names
    MnUserParameters upar;
    upar.add("mean", mean, 0.1);
    upar.add("sigma", rms, 0.1);
    upar.add("area", area, 0.1);

    // create Migrad minimizer
    MnMigrad migrad(theFCN, upar);

    // minimize
    FunctionMinimum min = migrad();

    // create MINOS error factory
    MnMinos minos(theFCN, min);

    {
      // 1-sigma MINOS errors (minimal interface)
      std::pair<double,double> e0 = minos(0);
      std::pair<double,double> e1 = minos(1);
      std::pair<double,double> e2 = minos(2);
      
      // output
      std::cout<<"1-sigma minos errors: "<<std::endl;
      std::cout<<"par0: "<<min.userState().value("mean")<<" "<<e0.first<<" "<<e0.second<<std::endl;
      std::cout<<"par1: "<<min.userState().value(1)<<" "<<e1.first<<" "<<e1.second<<std::endl;
      std::cout<<"par2: "<<min.userState().value("area")<<" "<<e2.first<<" "<<e2.second<<std::endl;
    }

    {
      // 2-sigma MINOS errors (rich interface)
      theFCN.setErrorDef(4.);
      MinosError e0 = minos.minos(0);
      MinosError e1 = minos.minos(1);
      MinosError e2 = minos.minos(2);
      
      // output
      std::cout<<"2-sigma minos errors: "<<std::endl;
      std::cout<<e0<<std::endl;
      std::cout<<e1<<std::endl;
      std::cout<<e2<<std::endl;
    }
  }

  {
    // demonstrate how to use the CONTOURs

    // create Minuit parameters with names
    MnUserParameters upar;
    upar.add("mean", mean, 0.1);
    upar.add("sigma", rms, 0.1);
    upar.add("area", area, 0.1);

    // create Migrad minimizer
    MnMigrad migrad(theFCN, upar);

    // minimize
    FunctionMinimum min = migrad();

    // create contours factory with FCN and minimum
    MnContours contours(theFCN, min);
  
    //70% confidence level for 2 parameters contour around the minimum
    // (minimal interface)
    theFCN.setErrorDef(2.41);
    std::vector<std::pair<double,double> > cont = contours(0, 1, 20);

    //95% confidence level for 2 parameters contour
    // (rich interface)
    theFCN.setErrorDef(5.99);
    ContoursError cont4 = contours.contour(0, 1, 20);
    
    // plot the contours
    MnPlot plot;
    cont.insert(cont.end(), cont4().begin(), cont4().end());
    plot(min.userState().value("mean"), min.userState().value("sigma"), cont);

    // print out one contour
    std::cout<<cont4<<std::endl;
  }

  return 0;
}
