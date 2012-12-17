#include "Quad1F.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnPrint.h"
#include "Minuit/VariableMetricMinimizer.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnMinos.h"

int main() {

  {
    //test constructor
    {
      Quad1F fcn;
      MnUserParameters upar;
      upar.add("x", 1., 0.1);
      MnMigrad migrad(fcn, upar);
      FunctionMinimum min = migrad();
      std::cout<<"min= "<<min<<std::endl;
    }
    {
      // using VariableMetricMinimizer, analytical derivatives
      Quad1F fcn;
      std::vector<double> par(1, 1.);
      std::vector<double> err(1, 0.1);
      VariableMetricMinimizer mini;
      FunctionMinimum min = mini.minimize(fcn, par, err);
      std::cout<<"min= "<<min<<std::endl;
    }
    {
      // test Minos for one parameter
      Quad1F fcn;
      std::vector<double> par(1, 1.);
      std::vector<double> err(1, 0.1);
      VariableMetricMinimizer mini;
      FunctionMinimum min = mini.minimize(fcn, par, err);
      MnMinos minos(fcn, min);
      std::pair<double,double> e0 = minos(0);
      std::cout<<"par0: "<<min.userState().value(unsigned(0))<<" "<<e0.first<<" "<<e0.second<<std::endl;
      fcn.setErrorDef(4.);
      MnMinos minos2(fcn, min);
      std::pair<double,double> e02 = minos2(0);
      std::cout<<"par0: "<<min.userState().value(unsigned(0))<<" "<<e02.first<<" "<<e02.second<<std::endl;
    }

  }

  return 0;
}
