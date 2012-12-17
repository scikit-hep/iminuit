#include "Minuit/MnMinos.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/FCNBase.h"
#include "Minuit/MnFunctionCross.h"
#include "Minuit/MnCross.h"
#include "Minuit/MinosError.h"
#include "Minuit/MnPrint.h"

std::pair<double,double> MnMinos::operator()(unsigned int par, unsigned int maxcalls) const {
  MinosError mnerr = minos(par, maxcalls);
  return mnerr();
}

double MnMinos::lower(unsigned int par, unsigned int maxcalls) const {

  MnUserParameterState upar = theMinimum.userState();
  double err = theMinimum.userState().error(par);

  MnCross aopt = loval(par, maxcalls);

  double lower = aopt.isValid() ? -1.*err*(1.+ aopt.value()) : (aopt.atLimit() ? upar.parameter(par).lowerLimit() : upar.value(par));
  
  return lower;
}

double MnMinos::upper(unsigned int par, unsigned int maxcalls) const {
  MnCross aopt = upval(par, maxcalls);
  
  MnUserParameterState upar = theMinimum.userState();
  double err = theMinimum.userState().error(par);

  double upper = aopt.isValid() ? err*(1.+ aopt.value()) : (aopt.atLimit() ? upar.parameter(par).upperLimit() : upar.value(par));
  
  return upper;
}

MinosError MnMinos::minos(unsigned int par, unsigned int maxcalls) const {
  assert(theMinimum.isValid());  
  assert(!theMinimum.userState().parameter(par).isFixed());
  assert(!theMinimum.userState().parameter(par).isConst());

  MnCross up = upval(par, maxcalls);
  MnCross lo = loval(par, maxcalls);

  return MinosError(par, theMinimum.userState().value(par), lo, up);
}

MnCross MnMinos::upval(unsigned int par, unsigned int maxcalls) const {
  assert(theMinimum.isValid());  
  assert(!theMinimum.userState().parameter(par).isFixed());
  assert(!theMinimum.userState().parameter(par).isConst());
  if(maxcalls == 0) {
    unsigned int nvar = theMinimum.userState().variableParameters();
    maxcalls = 2*(nvar+1)*(200 + 100*nvar + 5*nvar*nvar);
  }

  std::vector<unsigned int> para(1, par);

  MnUserParameterState upar = theMinimum.userState();
  double err = upar.error(par);
  double val = upar.value(par) + err;
  std::vector<double> xmid(1, val);
  std::vector<double> xdir(1, err);
  
  double up = theFCN.up();
  unsigned int ind = upar.intOfExt(par);
  MnAlgebraicSymMatrix m = theMinimum.error().matrix();
  double xunit = sqrt(up/err);
  for(unsigned int i = 0; i < m.nrow(); i++) {
    if(i == ind) continue;
    double xdev = xunit*m(ind,i);
    unsigned int ext = upar.extOfInt(i);
    upar.setValue(ext, upar.value(ext) + xdev);
  }

  upar.fix(par);
  upar.setValue(par, val);

//   double edmmax = 0.5*0.1*theFCN.up()*1.e-3;
  double toler = 0.1;
  MnFunctionCross cross(theFCN, upar, theMinimum.fval(), theStrategy);

  MnCross aopt = cross(para, xmid, xdir, toler, maxcalls);

//   std::cout<<"aopt= "<<aopt.value()<<std::endl;

  if(aopt.atLimit()) 
    std::cout<<"MnMinos parameter "<<par<<" is at upper limit."<<std::endl;
  if(aopt.atMaxFcn())
    std::cout<<"MnMinos maximum number of function calls exceeded for parameter "<<par<<std::endl;   
  if(aopt.newMinimum())
    std::cout<<"MnMinos new minimum found while looking for parameter "<<par<<std::endl;     
  if(!aopt.isValid()) 
    std::cout<<"MnMinos could not find upper value for parameter "<<par<<"."<<std::endl;

  return aopt;
}

MnCross MnMinos::loval(unsigned int par, unsigned int maxcalls) const {
  assert(theMinimum.isValid());  
  assert(!theMinimum.userState().parameter(par).isFixed());
  assert(!theMinimum.userState().parameter(par).isConst());
  if(maxcalls == 0) {
    unsigned int nvar = theMinimum.userState().variableParameters();
    maxcalls = 2*(nvar+1)*(200 + 100*nvar + 5*nvar*nvar);
  }
  std::vector<unsigned int> para(1, par);

  MnUserParameterState upar = theMinimum.userState();
  double err = upar.error(par);
  double val = upar.value(par) - err;
  std::vector<double> xmid(1, val);
  std::vector<double> xdir(1, -err);
  
  double up = theFCN.up();
  unsigned int ind = upar.intOfExt(par);
  MnAlgebraicSymMatrix m = theMinimum.error().matrix();
  double xunit = sqrt(up/err);
  for(unsigned int i = 0; i < m.nrow(); i++) {
    if(i == ind) continue;
    double xdev = xunit*m(ind,i);
    unsigned int ext = upar.extOfInt(i);
    upar.setValue(ext, upar.value(ext) - xdev);
  }

  upar.fix(par);
  upar.setValue(par, val);

//   double edmmax = 0.5*0.1*theFCN.up()*1.e-3;
  double toler = 0.1;
  MnFunctionCross cross(theFCN, upar, theMinimum.fval(), theStrategy);

  MnCross aopt = cross(para, xmid, xdir, toler, maxcalls);

//   std::cout<<"aopt= "<<aopt.value()<<std::endl;

  if(aopt.atLimit()) 
    std::cout<<"MnMinos parameter "<<par<<" is at lower limit."<<std::endl;
  if(aopt.atMaxFcn())
    std::cout<<"MnMinos maximum number of function calls exceeded for parameter "<<par<<std::endl;   
  if(aopt.newMinimum())
    std::cout<<"MnMinos new minimum found while looking for parameter "<<par<<std::endl;     
  if(!aopt.isValid()) 
    std::cout<<"MnMinos could not find lower value for parameter "<<par<<"."<<std::endl;

  return aopt;
}

