#include "Minuit/Numerical2PGradientCalculator.h"
#include "Minuit/InitialGradientCalculator.h"
#include "Minuit/MnFcn.h"
#include "Minuit/MnUserTransformation.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MinimumParameters.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MnStrategy.h"
#ifdef DEBUG
#include "Minuit/MnPrint.h"
#endif

#include <math.h>

FunctionGradient Numerical2PGradientCalculator::operator()(const MinimumParameters& par) const {

  InitialGradientCalculator gc(theFcn, theTransformation, theStrategy);
  FunctionGradient gra = gc(par);

  return (*this)(par, gra);  
}


// comment it, because it was added
FunctionGradient Numerical2PGradientCalculator::operator()(const std::vector<double>& params) const {

  int npar = params.size();
 
  MnAlgebraicVector par(npar);
  for (int i = 0; i < npar; ++i) {
    par(i) = params[i];
  }

  double fval = fcn()(par);

  MinimumParameters minpars = MinimumParameters(par, fval);

  return (*this)(minpars);

}



FunctionGradient Numerical2PGradientCalculator::operator()(const MinimumParameters& par, const FunctionGradient& gradient) const {

//    std::cout<<"########### Numerical2PDerivative"<<std::endl;
//    std::cout<<"initial grd: "<<gradient.grad()<<std::endl;
//    std::cout<<"position: "<<par.vec()<<std::endl;

  assert(par.isValid());

  MnAlgebraicVector x = par.vec();

  double fcnmin = par.fval();
//   std::cout<<"fval: "<<fcnmin<<std::endl;

  double dfmin = 8.*precision().eps2()*(fabs(fcnmin)+fcn().up());
  double vrysml = 8.*precision().eps()*precision().eps();
//   double vrysml = std::max(1.e-4, precision().eps2());
//    std::cout<<"dfmin= "<<dfmin<<std::endl;
//    std::cout<<"vrysml= "<<vrysml<<std::endl;
//    std::cout << " ncycle " << ncycle() << std::endl;
  
  unsigned int n = x.size();
//   MnAlgebraicVector vgrd(n), vgrd2(n), vgstp(n);
  MnAlgebraicVector grd = gradient.grad();
  MnAlgebraicVector g2 = gradient.g2();
  MnAlgebraicVector gstep = gradient.gstep();
  for(unsigned int i = 0; i < n; i++) {
    double xtf = x(i);
    double epspri = precision().eps2() + fabs(grd(i)*precision().eps2());
    double stepb4 = 0.;
    for(unsigned int j = 0; j < ncycle(); j++)  {
      double optstp = sqrt(dfmin/(fabs(g2(i))+epspri));
      double step = std::max(optstp, fabs(0.1*gstep(i)));
//       std::cout<<"step: "<<step;
      if(trafo().parameter(trafo().extOfInt(i)).hasLimits()) {
	if(step > 0.5) step = 0.5;
      }
      double stpmax = 10.*fabs(gstep(i));
      if(step > stpmax) step = stpmax;
//       std::cout<<" "<<step;
      double stpmin = std::max(vrysml, 8.*fabs(precision().eps2()*x(i)));
      if(step < stpmin) step = stpmin;
//       std::cout<<" "<<step<<std::endl;
//       std::cout<<"step: "<<step<<std::endl;
      if(fabs((step-stepb4)/step) < stepTolerance()) {
//  	std::cout<<"(step-stepb4)/step"<<std::endl;
//  	std::cout<<"j= "<<j<<std::endl;
//  	std::cout<<"step= "<<step<<std::endl;
	break;
      }
      gstep(i) = step;
      stepb4 = step;
//       MnAlgebraicVector pstep(n);
//       pstep(i) = step;
//       double fs1 = fcn()(pstate + pstep);
//       double fs2 = fcn()(pstate - pstep);

      x(i) = xtf + step;
      double fs1 = fcn()(x);
      x(i) = xtf - step;
      double fs2 = fcn()(x);
      x(i) = xtf;

      double grdb4 = grd(i);
      
      grd(i) = 0.5*(fs1 - fs2)/step;
      g2(i) = (fs1 + fs2 - 2.*fcnmin)/step/step;
      
      if(fabs(grdb4-grd(i))/(fabs(grd(i))+dfmin/step) < gradTolerance())  {
//  	std::cout<<"j= "<<j<<std::endl;
//  	std::cout<<"step= "<<step<<std::endl;
//  	std::cout<<"fs1, fs2: "<<fs1<<" "<<fs2<<std::endl;
//  	std::cout<<"fs1-fs2: "<<fs1-fs2<<std::endl;
	break;
      }
    }
    
//     vgrd(i) = grd;
//     vgrd2(i) = g2;
//     vgstp(i) = gstep;
  }  
//   std::cout<<"final grd: "<<grd<<std::endl;
//   std::cout<<"########### return from Numerical2PDerivative"<<std::endl;
  return FunctionGradient(grd, g2, gstep);
}

const MnMachinePrecision& Numerical2PGradientCalculator::precision() const {
  return theTransformation.precision();
}

unsigned int Numerical2PGradientCalculator::ncycle() const {
  return strategy().gradientNCycles();
}

double Numerical2PGradientCalculator::stepTolerance() const {
  return strategy().gradientStepTolerance();
}

double Numerical2PGradientCalculator::gradTolerance() const {
  return strategy().gradientTolerance();
}

