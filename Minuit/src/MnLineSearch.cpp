#include "Minuit/MnLineSearch.h"
#include "Minuit/MnFcn.h"
#include "Minuit/MinimumParameters.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MnParabola.h"
#include "Minuit/MnParabolaPoint.h"
#include "Minuit/MnParabolaFactory.h"
#include "Minuit/LaSum.h"

/**  Perform a line search from position defined by the vector st
       along the direction step, where the length of vector step
       gives the expected position of minimum.
       fcn is value of function at the starting position ,  
       gdel (if non-zero) is df/dx along step at st. 
       Return a parabola point containing minimum x posiiton and y (function value)
*/

MnParabolaPoint MnLineSearch::operator()(const MnFcn& fcn, const MinimumParameters& st, const MnAlgebraicVector& step, double gdel, const MnMachinePrecision& prec) const {

//   std::cout<<"gdel= "<<gdel<<std::endl;
//   std::cout<<"step= "<<step<<std::endl;

  double overal = 1000.;
  double undral = -100.;
  double toler = 0.05;
  double slamin = 0.;
  double slambg = 5.;
  double alpha = 2.;
  int maxiter = 12;
  int niter = 0;

  for(unsigned int i = 0; i < step.size(); i++) {
    if(fabs(step(i)) < prec.eps() )  continue;
    double ratio = fabs(st.vec()(i)/step(i));
    if(fabs(slamin) < prec.eps()) slamin = ratio;
    if(ratio < slamin) slamin = ratio;
  }
  if(fabs(slamin) < prec.eps()) slamin = prec.eps();
  slamin *= prec.eps2();

  double F0 = st.fval();
  double F1 = fcn(st.vec()+step);
  double fvmin = st.fval();
  double xvmin = 0.;

  if(F1 < F0) {
    fvmin = F1;
    xvmin = 1.;
  }
  double toler8 = toler;
  double slamax = slambg;
  double flast = F1;
  double slam = 1.;

  bool iterate = false;
  MnParabolaPoint p0(0., F0);
  MnParabolaPoint p1(slam, flast);
  double F2 = 0.;
  do {  
    // cut toler8 as function goes up 
    iterate = false;
    MnParabola pb = MnParabolaFactory()(p0, gdel, p1);
//     std::cout<<"pb.min() = "<<pb.min()<<std::endl;
//     std::cout<<"flast, F0= "<<flast<<", "<<F0<<std::endl;
//     std::cout<<"flast-F0= "<<flast-F0<<std::endl;
//     std::cout<<"slam= "<<slam<<std::endl;
//     double df = flast-F0;
//     if(fabs(df) < prec.eps2()) {
//       if(flast-F0 < 0.) df = -prec.eps2();
//       else df = prec.eps2();
//     }
//     std::cout<<"df= "<<df<<std::endl;
//     double denom = 2.*(df-gdel*slam)/(slam*slam);
    double denom = 2.*(flast-F0-gdel*slam)/(slam*slam);
//     std::cout<<"denom= "<<denom<<std::endl;
     if(fabs(denom) < prec.eps()) {
      denom = -0.1*gdel;
      slam = 1.;
    }
//     std::cout<<"slam= "<<slam<<std::endl;
    if(fabs(denom) > prec.eps()) slam = -gdel/denom;
//     std::cout<<"slam= "<<slam<<std::endl;
    if(slam < 0.) slam = slamax;
//     std::cout<<"slam= "<<slam<<std::endl;
    if(slam > slamax) slam = slamax;
//     std::cout<<"slam= "<<slam<<std::endl;
    if(slam < toler8) slam = toler8;
//     std::cout<<"slam= "<<slam<<std::endl;
    if(slam < slamin) {
//       std::cout<<"F1, F2= "<<p0.y()<<", "<<p1.y()<<std::endl;
//       std::cout<<"x1, x2= "<<p0.x()<<", "<<p1.x()<<std::endl;
//       std::cout<<"x, f= "<<xvmin<<", "<<fvmin<<std::endl;
      return MnParabolaPoint(xvmin, fvmin);
    }
    if(fabs(slam - 1.) < toler8 && p1.y() < p0.y()) {
//       std::cout<<"F1, F2= "<<p0.y()<<", "<<p1.y()<<std::endl;
//       std::cout<<"x1, x2= "<<p0.x()<<", "<<p1.x()<<std::endl;
//       std::cout<<"x, f= "<<xvmin<<", "<<fvmin<<std::endl;
      return MnParabolaPoint(xvmin, fvmin);
    }
    if(fabs(slam - 1.) < toler8) slam = 1. + toler8;

//     if(fabs(gdel) < prec.eps2() && fabs(denom) < prec.eps2())
//       slam = 1000.;
//     MnAlgebraicVector tmp = step;
//     tmp *= slam;
//     F2 = fcn(st.vec()+tmp);
    F2 = fcn(st.vec() + slam*step);
    if(F2 < fvmin) {
      fvmin = F2;
      xvmin = slam;
    }
    // LM : correct a bug using precision
    if (fabs( p0.y() - fvmin) < fabs(fvmin)*prec.eps() ) { 
      //   if(p0.y()-prec.eps() < fvmin && fvmin < p0.y()+prec.eps()) {
      iterate = true;
      flast = F2;
      toler8 = toler*slam;
      overal = slam - toler8;
      slamax = overal;
      p1 = MnParabolaPoint(slam, flast);
      niter++;
    }
  } while(iterate && niter < maxiter);
  if(niter >= maxiter) {
    // exhausted max number of iterations
    return MnParabolaPoint(xvmin, fvmin);  
  }
    
//   std::cout<<"after initial 2-point iter: "<<std::endl;
//   std::cout<<"F0, F1, F2= "<<p0.y()<<", "<<p1.y()<<", "<<F2<<std::endl;
//   std::cout<<"x0, x1, x2= "<<p0.x()<<", "<<p1.x()<<", "<<slam<<std::endl;

  MnParabolaPoint p2(slam, F2);
 
  do {
    slamax = std::max(slamax, alpha*fabs(xvmin));
    MnParabola pb = MnParabolaFactory()(p0, p1, p2);
//     std::cout<<"p2-p0,p1: "<<p2.y() - p0.y()<<", "<<p2.y() - p1.y()<<std::endl;
//     std::cout<<"a, b, c= "<<pb.a()<<" "<<pb.b()<<" "<<pb.c()<<std::endl;
    if(pb.a() < prec.eps2()) {
      double slopem = 2.*pb.a()*xvmin + pb.b();
      if(slopem < 0.) slam = xvmin + slamax;
      else slam = xvmin - slamax;
    } else {
      slam = pb.min();
 //      std::cout<<"pb.min() slam= "<<slam<<std::endl;
      if(slam > xvmin + slamax) slam = xvmin + slamax;
      if(slam < xvmin - slamax) slam = xvmin - slamax;
    }
    if(slam > 0.) {
      if(slam > overal) slam = overal;
    } else {
      if(slam < undral) slam = undral;
    }
//     std::cout<<" slam= "<<slam<<std::endl;

    double F3 = 0.;
    do {
      iterate = false;
      double toler9 = std::max(toler8, fabs(toler8*slam));
      // min. of parabola at one point    
      if(fabs(p0.x() - slam) < toler9 || 
	 fabs(p1.x() - slam) < toler9 || 
	 fabs(p2.x() - slam) < toler9) {
//   	std::cout<<"F1, F2, F3= "<<p0.y()<<", "<<p1.y()<<", "<<p2.y()<<std::endl;
//   	std::cout<<"x1, x2, x3= "<<p0.x()<<", "<<p1.x()<<", "<<p2.x()<<std::endl;
//   	std::cout<<"x, f= "<<xvmin<<", "<<fvmin<<std::endl;
	return MnParabolaPoint(xvmin, fvmin);
      }

      // take the step
//       MnAlgebraicVector tmp = step;
//       tmp *= slam;
      F3 = fcn(st.vec() + slam*step);
//       std::cout<<"F3= "<<F3<<std::endl;
//       std::cout<<"F3-p(2-0).y()= "<<F3-p2.y()<<" "<<F3-p1.y()<<" "<<F3-p0.y()<<std::endl;
      // if latest point worse than all three previous, cut step
      if(F3 > p0.y() && F3 > p1.y() && F3 > p2.y()) {
//   	std::cout<<"F3 worse than all three previous"<<std::endl;
	if(slam > xvmin) overal = std::min(overal, slam-toler8);
	if(slam < xvmin) undral = std::max(undral, slam+toler8);	
	slam = 0.5*(slam + xvmin);
//   	std::cout<<"new slam= "<<slam<<std::endl;
	iterate = true;
	niter++;
      }
    } while(iterate && niter < maxiter);
    if(niter >= maxiter) {
      // exhausted max number of iterations
      return MnParabolaPoint(xvmin, fvmin);  
    }

    // find worst previous point out of three and replace
    MnParabolaPoint p3(slam, F3);
    if(p0.y() > p1.y() && p0.y() > p2.y()) p0 = p3;
    else if(p1.y() > p0.y() && p1.y() > p2.y()) p1 = p3;
    else p2 = p3;
    if(F3 < fvmin) {
      fvmin = F3;
      xvmin = slam;
    } else {
      if(slam > xvmin) overal = std::min(overal, slam-toler8);
      if(slam < xvmin) undral = std::max(undral, slam+toler8);	
    }
    
    niter++;
  } while(niter < maxiter);

//   std::cout<<"F1, F2= "<<p0.y()<<", "<<p1.y()<<std::endl;
//   std::cout<<"x1, x2= "<<p0.x()<<", "<<p1.x()<<std::endl;
//   std::cout<<"x, f= "<<xvmin<<", "<<fvmin<<std::endl;
  return MnParabolaPoint(xvmin, fvmin);
}
