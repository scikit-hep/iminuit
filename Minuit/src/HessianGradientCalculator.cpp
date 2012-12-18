#include "Minuit/HessianGradientCalculator.h"
#include "Minuit/InitialGradientCalculator.h"
#include "Minuit/MnFcn.h"
#include "Minuit/MnUserTransformation.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MinimumParameters.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MnStrategy.h"

#include <math.h>

FunctionGradient HessianGradientCalculator::operator()(const MinimumParameters& par) const {

    InitialGradientCalculator gc(theFcn, theTransformation, theStrategy);
    FunctionGradient gra = gc(par);

    return (*this)(par, gra);
}

FunctionGradient HessianGradientCalculator::operator()(const MinimumParameters& par,
    const FunctionGradient& gradient) const {
    std::pair<FunctionGradient, MnAlgebraicVector> mypair = deltaGradient(par, gradient);

    return mypair.first;
}

const MnMachinePrecision& HessianGradientCalculator::precision() const {
    return theTransformation.precision();
}

unsigned int HessianGradientCalculator::ncycle() const {
    return strategy().hessianGradientNCycles();
}

double HessianGradientCalculator::stepTolerance() const {
    return strategy().gradientStepTolerance();
}

double HessianGradientCalculator::gradTolerance() const {
    return strategy().gradientTolerance();
}

std::pair<FunctionGradient, MnAlgebraicVector> HessianGradientCalculator::deltaGradient(
    const MinimumParameters& par, const FunctionGradient& gradient) const {

    assert(par.isValid());

    MnAlgebraicVector x = par.vec();
    MnAlgebraicVector grd = gradient.grad();
    const MnAlgebraicVector& g2 = gradient.g2();
    const MnAlgebraicVector& gstep = gradient.gstep();

    double fcnmin = par.fval();
//   std::cout<<"fval: "<<fcnmin<<std::endl;

    double dfmin = 4.*precision().eps2()*(fabs(fcnmin)+fcn().up());

    unsigned int n = x.size();
    MnAlgebraicVector dgrd(n);

// initial starting values
    for(unsigned int i = 0; i < n; i++) {
        double xtf = x(i);
        double dmin = 4.*precision().eps2()*(xtf + precision().eps2());
        double epspri = precision().eps2() + fabs(grd(i)*precision().eps2());
        double optstp = sqrt(dfmin/(fabs(g2(i))+epspri));
        double d = 0.2*fabs(gstep(i));
        if(d > optstp) d = optstp;
        if(d < dmin) d = dmin;
        double chgold = 10000.;
        double dgmin = 0.;
        double grdold = 0.;
        double grdnew = 0.;
        for(unsigned int j = 0; j < ncycle(); j++)  {
            x(i) = xtf + d;
            double fs1 = fcn()(x);
            x(i) = xtf - d;
            double fs2 = fcn()(x);
            x(i) = xtf;
            //double sag = 0.5*(fs1+fs2-2.*fcnmin);
            grdold = grd(i);
            grdnew = (fs1-fs2)/(2.*d);
            dgmin = precision().eps()*(fabs(fs1) + fabs(fs2))/d;
            if(fabs(grdnew) < precision().eps()) break;
            double change = fabs((grdold-grdnew)/grdnew);
            if(change > chgold && j > 1) break;
            chgold = change;
            grd(i) = grdnew;
            if(change < 0.05) break;
            if(fabs(grdold-grdnew) < dgmin) break;
            if(d < dmin) break;
            d *= 0.2;
        }
        dgrd(i) = std::max(dgmin, fabs(grdold-grdnew));
    }

    return std::pair<FunctionGradient, MnAlgebraicVector>(FunctionGradient(grd, g2, gstep), dgrd);
}
