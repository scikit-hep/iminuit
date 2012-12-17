#include "Minuit/MnScan.h"
#include "Minuit/MnParameterScan.h"

std::vector<std::pair<double, double> > MnScan::scan(unsigned int par, unsigned int maxsteps, double low, double high) {
 
  MnParameterScan scan(theFCN, theState.parameters());
  double amin = scan.fval();

  std::vector<std::pair<double, double> > result = scan(par, maxsteps, low, high);
  if(scan.fval() < amin) {
    theState.setValue(par, scan.parameters().value(par));
    amin = scan.fval();
  }
   
  return result;
}
