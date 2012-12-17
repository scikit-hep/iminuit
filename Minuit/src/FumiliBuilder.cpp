#include "Minuit/FumiliBuilder.h"
#include "Minuit/FumiliStandardMaximumLikelihoodFCN.h"
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

//#define DEBUG 1


double inner_product(const LAVector&, const LAVector&);

FunctionMinimum FumiliBuilder::minimum(const MnFcn& fcn, const GradientCalculator& gc, const MinimumSeed& seed, const MnStrategy& strategy, unsigned int maxfcn, double edmval) const {


  edmval *= 0.0001;


#ifdef DEBUG
  std::cout<<"FumiliBuilder convergence when edm < "<<edmval<<std::endl;
#endif

  if(seed.parameters().vec().size() == 0) {
    return FunctionMinimum(seed, fcn.up());
  }


//   double edm = estimator().estimate(seed.gradient(), seed.error());
  double edm = seed.state().edm();

  FunctionMinimum min(seed, fcn.up() );

  if(edm < 0.) {
    std::cout<<"FumiliBuilder: initial matrix not pos.def."<<std::endl;
    //assert(!seed.error().isPosDef());
    return min;
  }

  std::vector<MinimumState> result;
//   result.reserve(1);
  result.reserve(8);

  result.push_back( seed.state() );

  // do actual iterations


  // try first with a maxfxn = 50% of maxfcn 
  // FUmili in principle needs much less iterations
  int maxfcn_eff = int(0.5*maxfcn);
  int ipass = 0;
  double edmprev = 1;
  
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

      // break the loop if edm is NOT getting smaller 
      if (ipass > 0 && edm >= edmprev) { 
	std::cout << "FumiliBuilder: Exit iterations, no improvements after Hesse. edm is  " << edm << " previous " << edmprev << std::endl;
	break; 
      } 
      if (edm > edmval) { 
#ifdef DEBUG
	std::cout << "FumiliBuilder: Tolerance is not sufficient - edm is " << edm << " requested " << edmval 
		  << " continue the minimization" << std::endl;
#endif
      }
    }

    // end loop on iterations
    // ? need a maximum here (or max of function calls is enough ? ) 
    // continnue iteration (re-calculate funciton minimum if edm IS NOT sufficient) 
    // no need to check that hesse calculation is done (if isnot done edm is OK anyway)
    // count the pass to exit second time when function minimum is invalid
    // increase by 20% maxfcn for doing some more tests
    if (ipass == 0) maxfcn_eff = maxfcn;
    ipass++;
    edmprev = edm; 

  }  while (edm > edmval );



  // add latest state (hessian calculation)
  min.add( result.back() );

  return min;

}

FunctionMinimum FumiliBuilder::minimum(const MnFcn& fcn, const GradientCalculator& gc, const MinimumSeed& seed, std::vector<MinimumState>& result, unsigned int maxfcn, double edmval) const {



  /*
    Three options were possible:
    
    1) create two parallel and completely separate hierarchies, in which case
    the FumiliMinimizer would NOT inherit from ModularFunctionMinimizer, 
    FumiliBuilder would not inherit from MinimumBuilder etc

    2) Use the inheritance (base classes of ModularFunctionMinimizer,
    MinimumBuilder etc), but recreate the member functions minimize() and 
    minimum() respectively (naming them for example minimize2() and 
    minimum2()) so that they can take FumiliFCNBase as parameter instead FCNBase
    (otherwise one wouldn't be able to call the Fumili-specific methods).

    3) Cast in the daughter classes derived from ModularFunctionMinimizer,
    MinimumBuilder.

    The first two would mean to duplicate all the functionality already existent,
    which is a very bad practice and error-prone. The third one is the most
    elegant and effective solution, where the only constraint is that the user
    must know that he has to pass a subclass of FumiliFCNBase to the FumiliMinimizer 
    and not just a subclass of FCNBase.
    BTW, the first two solutions would have meant to recreate also a parallel
    structure for MnFcn...
  **/
  //  const FumiliFCNBase* tmpfcn =  dynamic_cast<const FumiliFCNBase*>(&(fcn.fcn()));

  const MnMachinePrecision& prec = seed.precision();

  const MinimumState & initialState = result.back();

  double edm = initialState.edm();


#ifdef DEBUG
  std::cout << "\n\nDEBUG FUMILI Builder  \nSEED State: "  
	    << " Parameter " << seed.state().vec()       
	    << " Gradient " << seed.gradient().vec() 
	    << " Inv Hessian " << seed.error().invHessian()  
	    << " edm = " << seed.state().edm() 
            << " maxfcn = " << maxfcn 
            << " tolerance = " << edmval 
	    << std::endl; 
#endif


  // iterate until edm is small enough or max # of iterations reached
  edm *= (1. + 3.*seed.error().dcovar());
  MnLineSearch lsearch;
  MnAlgebraicVector step(seed.gradient().vec().size());

  // initial lambda value
  double lambda = 0.001; 
  //double lambda = 0.0; 


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
      std::cout<<"FumiliBuilder: matrix not pos.def."<<std::endl;
      std::cout<<"gdel > 0: "<<gdel<<std::endl;
      MnPosDef psdf;
      s0 = psdf(s0, prec);
      step = -1.*s0.error().invHessian()*s0.gradient().vec();
      gdel = inner_product(step, s0.gradient().grad());
      std::cout<<"gdel: "<<gdel<<std::endl;
      if(gdel > 0.) {
	result.push_back(s0);
	return FunctionMinimum(seed, result, fcn.up());
      }
    }


//     MnParabolaPoint pp = lsearch(fcn, s0.parameters(), step, gdel, prec);

//     if(fabs(pp.y() - s0.fval()) < prec.eps()) {
//       std::cout<<"FumiliBuilder: no improvement"<<std::endl;
//       break; //no improvement
//     }


//     MinimumParameters p(s0.vec() + pp.x()*step, pp.y());

    // if taking a full step 

    // take a full step

    MinimumParameters p(s0.vec() + step,  fcn( s0.vec() + step ) );

#ifdef DEBUG
    std::cout << "Before gradient " << fcn.numOfCalls() << std::endl; 
#endif
        
    FunctionGradient g = gc(p, s0.gradient());
 
#ifdef DEBUG   
    std::cout << "After gradient " << fcn.numOfCalls() << std::endl; 
#endif

    //FunctionGradient g = gc(s0.parameters(), s0.gradient()); 


    // move error updator after gradient since the value is cached inside

    MinimumError e = errorUpdator().update(s0, p, gc, lambda);

    edm = estimator().estimate(g, s0.error());

    
#ifdef DEBUG
    std::cout << "Updated new point: \n " 
              << " Parameter " << p.vec()       
	      << " Gradient " << g.vec() 
	      << " InvHessian " << e.matrix() 
	      << " Hessian " << e.hessian() 
	      << " edm = " << edm << std::endl << std::endl;
#endif

    if(edm < 0.) {
      std::cout<<"FumiliBuilder: matrix not pos.def."<<std::endl;
      std::cout<<"edm < 0"<<std::endl;
      MnPosDef psdf;
      s0 = psdf(s0, prec);
      edm = estimator().estimate(g, s0.error());
      if(edm < 0.) {
	result.push_back(s0);
	return FunctionMinimum(seed, result, fcn.up());
      }
    } 
 
    // check lambda according to step 
    if ( p.fval() < s0.fval()  ) 
      // fcn is decreasing along the step
      lambda *= 0.1;
    else 
      lambda *= 10; 

#ifdef DEBUG
    std::cout <<  " finish iteration- " << result.size() << " lambda =  "  << lambda << " f1 = " << p.fval() << " f0 = " << s0.fval() << " num of calls = " << fcn.numOfCalls() << " edm " << edm << std::endl; 
#endif
  

    //std::cout << "FumiliBuilder DEBUG e.matrix()" << e.matrix() << std::endl;
    //std::cout << "DEBUG FumiliBuilder e.hessian()" << e.hessian() << std::endl;

    result.push_back(MinimumState(p, e, g, edm, fcn.numOfCalls())); 

    //std::cout << "FumiliBuilder DEBUG " << FunctionMinimum(seed, result, fcn.up()) << std::endl;
 

    edm *= (1. + 3.*e.dcovar());

    /**std::cout << "DEBUG FumiliBuilder edm: " << edm << " edmval: " << edmval <<
      " fcn.numOfCalls: " << fcn.numOfCalls() << " maxfcn: " << maxfcn << 
      " condition: " << (edm > edmval && fcn.numOfCalls() < maxfcn) <<std::endl;
    */


  } while(edm > edmval && fcn.numOfCalls() < maxfcn);
  


  if(fcn.numOfCalls() >= maxfcn) {
    std::cout<<"FumiliBuilder: call limit exceeded."<<std::endl;
    return FunctionMinimum(seed, result, fcn.up(), FunctionMinimum::MnReachedCallLimit());
  }

  if(edm > edmval) {
    if(edm < fabs(prec.eps2()*result.back().fval())) {
      std::cout<<"FumiliBuilder: machine accuracy limits further improvement."<<std::endl;
      return FunctionMinimum(seed, result, fcn.up());
    } else if(edm < 10.*edmval) {
      return FunctionMinimum(seed, result, fcn.up());
    } else {
      std::cout<<"FumiliBuilder: finishes without convergence."<<std::endl;
      std::cout<<"FumiliBuilder: edm= "<<edm<<" requested: "<<edmval<<std::endl;
      return FunctionMinimum(seed, result, fcn.up(), FunctionMinimum::MnAboveMaxEdm());
    }
  }
//   std::cout<<"result.back().error().dcovar()= "<<result.back().error().dcovar()<<std::endl;

#ifdef DEBUG
  std::cout << "Exiting succesfully FumiliBuilder \n" 
	    << "NFCalls = " << fcn.numOfCalls() 
	    << "\nFval = " <<  result.back().fval() 
	    << "\nedm = " << edm << " requested = " << edmval << std::endl; 
#endif

  return FunctionMinimum(seed, result, fcn.up());
}
