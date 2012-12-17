#include "Quad4F.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnUserParameters.h"
#include "Minuit/MnPrint.h"
// #include "TimingUtilities/PentiumTimer.h"

// StackAllocator gStackAllocator;

int main() {

  Quad4F fcn;

//   PentiumTimer stopwatch;
//   stopwatch.start();

//   long long int start = stopwatch.lap().ticks();
//   long long int stop = stopwatch.lap().ticks();
//   std::cout<<"stop-start: "<<stop - start<<std::endl;
//   start = stopwatch.lap().ticks();
  {
  //test constructor
  MnUserParameters upar;
  upar.add("x", 1., 0.1);
  upar.add("y", 1., 0.1);
  upar.add("z", 1., 0.1);
  upar.add("w", 1., 0.1);

  MnMigrad migrad(fcn, upar);
  FunctionMinimum min = migrad();
  std::cout<<"minimum: "<<min<<std::endl;
  }
//   stop = stopwatch.lap().ticks();
//   std::cout<<"stop-start: "<<stop - start<<std::endl;
/*

  {
  //test constructor
  std::vector<double> par(4); 
  std::vector<double> err(4);
  for(int i = 0; i < 4; i++) {
    par[i] = 1.;
    err[i] = 0.1;
  }
  MnMigrad migrad(fcn, par, err);
  FunctionMinimum min = migrad();
  std::cout<<"minimum: "<<min<<std::endl;
  }

  {
  //test edm value
  std::vector<double> par(4); 
  std::vector<double> err(4);
  for(int i = 0; i < 4; i++) {
    par[i] = 1.;
    err[i] = 0.1;
  }
  MnMigrad migrad(fcn, par, err);
  double edm = 1.e-1;
  FunctionMinimum min = migrad(20, edm);
  std::cout<<"minimum: "<<min<<std::endl;
  }

  {
  //test # of iterations
  std::vector<double> par(4); 
  std::vector<double> err(4);
  for(int i = 0; i < 4; i++) {
    par[i] = 1.;
    err[i] = 0.1;
  }
  MnMigrad migrad(fcn, par, err);
  int niter = 2;
  FunctionMinimum min = migrad(niter, 1.e-5);
  std::cout<<"minimum: "<<min<<std::endl;
  }
*/

  return 0;
}
