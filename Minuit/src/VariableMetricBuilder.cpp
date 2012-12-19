#include "Minuit/VariableMetricBuilder.h"
#include "Minuit/GradientCalculator.h"
#include "Minuit/MinimumState.h"
#include "Minuit/MinimumError.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnLineSearch.h"
#include "Minuit/MinimumSeed.h"
#include "Minuit/MnFcn.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MnPosDef.h"
#include "Minuit/MnParabolaPoint.h"
#include "Minuit/LaSum.h"
#include "Minuit/LaProd.h"
#include "Minuit/MnStrategy.h"
#include "Minuit/MnHesse.h"
#include "Minuit/MnPrint.h"
#include <iostream>
//#define DEBUG 0

#ifdef DEBUG
#include "Minuit/MnPrint.h"
#endif

double inner_product(const LAVector&, const LAVector&);

int VariableMetricBuilder::print_level = 1;

void VariableMetricBuilder::setPrintLevel(int p){
    VariableMetricBuilder::print_level = p;
}

FunctionMinimum VariableMetricBuilder::minimum(const MnFcn& fcn,
    const GradientCalculator& gc, const MinimumSeed& seed,
    const MnStrategy& strategy, unsigned int maxfcn, double edmval) const {
    using namespace std;
    edmval *= 0.0001;
    if(VariableMetricBuilder::print_level >= 1)
        cout << "======================================" << endl;

#ifdef DEBUG
    std::cout<<"VariableMetricBuilder convergence when edm < "<<edmval<<std::endl;
#endif

    if(seed.parameters().vec().size() == 0) {
        return FunctionMinimum(seed, fcn.up());
    }


//   double edm = estimator().estimate(seed.gradient(), seed.error());
    double edm = seed.state().edm();

    FunctionMinimum min(seed, fcn.up() );

    if(edm < 0.) {
        std::cout<<"VariableMetricBuilder: initial matrix not pos.def."<<std::endl;
        //assert(!seed.error().isPosDef());
        return min;
    }

    std::vector<MinimumState> result;
    //result.reserve(1);
    result.reserve(8);

    result.push_back( seed.state() );

    // do actual iterations


    // try first with a maxfxn = 80% of maxfcn
    int maxfcn_eff = maxfcn;
    int ipass = 0;

    do {


        min = minimum(fcn, gc, seed, result, maxfcn_eff, edmval);
        // second time check for validity of function minimum
        if (ipass > 0) {
            if(!min.isValid()) {
                std::cout<<"FunctionMinimum is invalid."<<std::endl;
                return min;
            }
        }

        // resulting edm of minimization
        edm = result.back().edm();

        if( (strategy.strategy() == 2) ||
            (strategy.strategy() == 1 && min.error().dcovar() > 0.05) ) {

#ifdef DEBUG
            std::cout<<"MnMigrad will verify convergence and error matrix. "<< std::endl;
            std::cout<<"dcov is =  "<<  min.error().dcovar() << std::endl;
#endif

            MinimumState st = MnHesse(strategy)(fcn, min.state(), min.seed().trafo());
            result.push_back( st );

            // check edm
            edm = st.edm();
#ifdef DEBUG
            std::cout << "edm after Hesse calculation " << edm << std::endl;
#endif
            if (edm > edmval) {
                std::cout << "VariableMetricBuilder: Tolerance is not sufficient - edm is " << edm << " requested " << edmval 
                << " continue the minimization" << std::endl;
            }
            min.add( result.back() );
        }

        // end loop on iterations
        // ? need a maximum here (or max of function calls is enough ? )
        // continnue iteration (re-calculate funciton minimum if edm IS NOT sufficient)
        // no need to check that hesse calculation is done (if isnot done edm is OK anyway)
        // count the pass to exit second time when function minimum is invalid
        // increase by 20% maxfcn for doing some more tests
        if (ipass == 0) maxfcn_eff = int(maxfcn*1.3);
        if(VariableMetricBuilder::print_level >= 1) min.print();
        ipass++;
    }  while (edm > edmval );
    //add hessian calculation back
    min.add( result.back() );
    return min;
}

FunctionMinimum VariableMetricBuilder::minimum(const MnFcn& fcn,
    const GradientCalculator& gc, const MinimumSeed& seed,
    std::vector<MinimumState>& result, unsigned int maxfcn, double edmval) const {


    const MnMachinePrecision& prec = seed.precision();


//   result.push_back(MinimumState(seed.parameters(), seed.error(), seed.gradient(), edm, fcn.numOfCalls()));
    const MinimumState & initialState = result.back();


    double edm = initialState.edm();


#ifdef DEBUG
    std::cout << "\n\nDEBUG Variable Metric Builder  \nInitial State: "
    << " Parameter " << initialState.vec()
    << " Gradient " << initialState.gradient().vec()
    << " Inv Hessian " << initialState.error().invHessian()
    << " edm = " << initialState.edm() << std::endl;
#endif



// iterate until edm is small enough or max # of iterations reached
    edm *= (1. + 3.*initialState.error().dcovar());
    MnLineSearch lsearch;
    MnAlgebraicVector step(initialState.gradient().vec().size());
// keep also prevStep
    MnAlgebraicVector prevStep(initialState.gradient().vec().size());

    do {

//     const MinimumState& s0 = result.back();
        MinimumState s0 = result.back();

        step = -1.*s0.error().invHessian()*s0.gradient().vec();

#ifdef DEBUG
        std::cout << "\n\n---> Iteration - " << result.size()
        << "\nFval = " << s0.fval() << " numOfCall = " << fcn.numOfCalls()
        << "\nInternal Parameter values " << s0.vec()
        << " Newton step " << step << std::endl;
#endif


        double gdel = inner_product(step, s0.gradient().grad());
        if(gdel > 0.) {
            std::cout<<"VariableMetricBuilder: matrix not pos.def."<<std::endl;
            std::cout<<"gdel > 0: "<<gdel<<std::endl;
            MnPosDef psdf;
            s0 = psdf(s0, prec);
            step = -1.*s0.error().invHessian()*s0.gradient().vec();
// #ifdef DEBUG
//       std::cout << "After MnPosdef - error  " << s0.error().invHessian() << " gradient " << s0.gradient().vec() << " step " << step << std::endl;
// #endif
            gdel = inner_product(step, s0.gradient().grad());
            std::cout<<"gdel: "<<gdel<<std::endl;
            if(gdel > 0.) {
                result.push_back(s0);
                return FunctionMinimum(seed, result, fcn.up());
            }
        }
        MnParabolaPoint pp = lsearch(fcn, s0.parameters(), step, gdel, prec);
        if(fabs(pp.y() - s0.fval()) < fabs(s0.fval())*prec.eps() ) {
            std::cout<<"VariableMetricBuilder: warning: no improvement in line search  " << std::endl;
// no improvement exit   (is it really needed LM ? in vers. 1.22 tried alternative )
            break;


        }

#ifdef DEBUG
        std::cout << "Result after line search : \nx = " << pp.x()
        << "\nOld Fval = " << s0.fval()
        << "\nNew Fval = " << pp.y()
        << "\nNFcalls = " << fcn.numOfCalls() << std::endl;
#endif

        MinimumParameters p(s0.vec() + pp.x()*step, pp.y());


        FunctionGradient g = gc(p, s0.gradient());


        edm = estimator().estimate(g, s0.error());


        if(edm < 0.) {
            std::cout<<"VariableMetricBuilder: matrix not pos.def."<<std::endl;
            std::cout<<"edm < 0"<<std::endl;
            MnPosDef psdf;
            s0 = psdf(s0, prec);
            edm = estimator().estimate(g, s0.error());
            if(edm < 0.) {
                result.push_back(s0);
                return FunctionMinimum(seed, result, fcn.up());
            }
        }
        MinimumError e = errorUpdator().update(s0, p, g);

#ifdef DEBUG
        std::cout << "Updated new point: \n "
        << " Parameter " << p.vec()
        << " Gradient " << g.vec()
        << " InvHessian " << e.matrix()
        << " Hessian " << e.hessian()
        << " edm = " << edm << std::endl << std::endl;
#endif


        result.push_back(MinimumState(p, e, g, edm, fcn.numOfCalls()));

        // correct edm
        edm *= (1. + 3.*e.dcovar());

#ifdef DEBUG
        std::cout << "edm corrected = " << edm << std::endl;
#endif
        FunctionMinimum tmp(seed, result, fcn.up());
        if(VariableMetricBuilder::print_level >= 2){
            tmp.print(true);
        }
    } while(edm > edmval && fcn.numOfCalls() < maxfcn);

    if(fcn.numOfCalls() >= maxfcn) {
        std::cout<<"VariableMetricBuilder: call limit exceeded."<<std::endl;
        return FunctionMinimum(seed, result, fcn.up(), FunctionMinimum::MnReachedCallLimit());
    }

    if(edm > edmval) {
        if(edm < fabs(prec.eps2()*result.back().fval())) {
            std::cout<<"VariableMetricBuilder: machine accuracy limits further improvement."<<std::endl;
            return FunctionMinimum(seed, result, fcn.up());
        } else if(edm < 10.*edmval) {
            return FunctionMinimum(seed, result, fcn.up());
        } else {
            std::cout<<"VariableMetricBuilder: finishes without convergence."<<std::endl;
            std::cout<<"VariableMetricBuilder: edm= "<<edm<<" requested: "<<edmval<<std::endl;
            return FunctionMinimum(seed, result, fcn.up(), FunctionMinimum::MnAboveMaxEdm());
        }
    }
    //std::cout<<"result.back().error().dcovar()= "<<result.back().error().dcovar()<<std::endl;

#ifdef DEBUG
    std::cout << "Exiting succesfully Variable Metric Builder \n"
    << "NFCalls = " << fcn.numOfCalls()
    << "\nFval = " <<  result.back().fval()
    << "\nedm = " << edm << " requested = " << edmval << std::endl;
#endif

    return FunctionMinimum(seed, result, fcn.up());
}
