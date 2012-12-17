#include "Minuit/MnHesse.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnUserFcn.h"
#include "Minuit/FCNBase.h"
#include "Minuit/MnPosDef.h"
#include "Minuit/HessianGradientCalculator.h"
#include "Minuit/Numerical2PGradientCalculator.h"
#include "Minuit/InitialGradientCalculator.h"
#include "Minuit/MinimumState.h"
#include "Minuit/VariableMetricEDMEstimator.h"
#include "Minuit/MnPrint.h"

MnUserParameterState MnHesse::operator()(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int maxcalls) const {
  return (*this)(fcn, MnUserParameterState(par, err), maxcalls);
}

MnUserParameterState MnHesse::operator()(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& cov, unsigned int nrow, unsigned int maxcalls) const {
  return (*this)(fcn, MnUserParameterState(par, cov, nrow), maxcalls);
}

MnUserParameterState MnHesse::operator()(const FCNBase& fcn, const std::vector<double>& par, const MnUserCovariance& cov, unsigned int maxcalls) const {
  return (*this)(fcn, MnUserParameterState(par, cov), maxcalls);
}

MnUserParameterState MnHesse::operator()(const FCNBase& fcn, const MnUserParameters& par, unsigned int maxcalls) const {
  return (*this)(fcn, MnUserParameterState(par), maxcalls);
}

MnUserParameterState MnHesse::operator()(const FCNBase& fcn, const MnUserParameters& par, const MnUserCovariance& cov, unsigned int maxcalls) const {
  return (*this)(fcn, MnUserParameterState(par, cov), maxcalls);
}

MnUserParameterState MnHesse::operator()(const FCNBase& fcn, const MnUserParameterState& state, unsigned int maxcalls) const {

  unsigned int n = state.variableParameters();
  MnUserFcn mfcn(fcn, state.trafo());
  MnAlgebraicVector x(n);
  for(unsigned int i = 0; i < n; i++) x(i) = state.intParameters()[i];
  double amin = mfcn(x);
  Numerical2PGradientCalculator gc(mfcn, state.trafo(), theStrategy);
  MinimumParameters par(x, amin);
  FunctionGradient gra = gc(par);
  MinimumState tmp = (*this)(mfcn, MinimumState(par, MinimumError(MnAlgebraicSymMatrix(n), 1.), gra, state.edm(), state.nfcn()), state.trafo(), maxcalls);

  return MnUserParameterState(tmp, fcn.up(), state.trafo());
}

MinimumState MnHesse::operator()(const MnFcn& mfcn, const MinimumState& st, const MnUserTransformation& trafo, unsigned int maxcalls) const {

  const MnMachinePrecision& prec = trafo.precision();
  // make sure starting at the right place
  double amin = mfcn(st.vec());
  double aimsag = sqrt(prec.eps2())*(fabs(amin)+mfcn.up());
  
  // diagonal elements first

  unsigned int n = st.parameters().vec().size();
  if(maxcalls == 0) maxcalls = 200 + 100*n + 5*n*n;

  MnAlgebraicSymMatrix vhmat(n);
  MnAlgebraicVector g2 = st.gradient().g2();
  MnAlgebraicVector gst = st.gradient().gstep();
  MnAlgebraicVector grd = st.gradient().grad();
  MnAlgebraicVector dirin = st.gradient().gstep();
  MnAlgebraicVector yy(n);
  if(st.gradient().isAnalytical()) {
    InitialGradientCalculator igc(mfcn, trafo, theStrategy);
    FunctionGradient tmp = igc(st.parameters());
    gst = tmp.gstep();
    dirin = tmp.gstep();
    g2 = tmp.g2();
  }

  MnAlgebraicVector x = st.parameters().vec(); 

  for(unsigned int i = 0; i < n; i++) {

    double xtf = x(i);
    double dmin = 8.*prec.eps2()*(fabs(xtf) + prec.eps2());
    double d = fabs(gst(i));
    if(d < dmin) d = dmin;

    for(unsigned int icyc = 0; icyc < ncycles(); icyc++) {
      double sag = 0.;
      double fs1 = 0.;
      double fs2 = 0.;
      for(unsigned int multpy = 0; multpy < 5; multpy++) {
	x(i) = xtf + d;
	fs1 = mfcn(x);
	x(i) = xtf - d;
	fs2 = mfcn(x);
	x(i) = xtf;
	sag = 0.5*(fs1+fs2-2.*amin);
	if(sag > prec.eps2()) goto L30; // break;
	if(trafo.parameter(i).hasLimits()) {
	  if(d > 0.5) goto L26;
	  d *= 10.;
	  if(d > 0.5) d = 0.51;
	  continue;
	}
	d *= 10.;
      }
      
L26:  
      std::cout<<"MnHesse: 2nd derivative zero for parameter "<<i<<std::endl;
      std::cout<<"MnHesse fails and will return diagonal matrix "<<std::endl;

      for(unsigned int j = 0; j < n; j++) {
	double tmp = g2(j) < prec.eps2() ? 1. : 1./g2(j);
	vhmat(j,j) = tmp < prec.eps2() ? 1. : tmp;
      }

      return MinimumState(st.parameters(), MinimumError(vhmat, MinimumError::MnHesseFailed()), st.gradient(), st.edm(), mfcn.numOfCalls());

L30:      
      double g2bfor = g2(i);
      g2(i) = 2.*sag/(d*d);
      grd(i) = (fs1-fs2)/(2.*d);
      gst(i) = d;
      dirin(i) = d;
      yy(i) = fs1;
      double dlast = d;
      d = sqrt(2.*aimsag/fabs(g2(i)));
      if(trafo.parameter(i).hasLimits()) d = std::min(0.5, d);
      if(d < dmin) d = dmin;

      // see if converged
      if(fabs((d-dlast)/d) < tolerstp()) break;
      if(fabs((g2(i)-g2bfor)/g2(i)) < tolerg2()) break; 
      d = std::min(d, 10.*dlast);
      d = std::max(d, 0.1*dlast);   
    }
    vhmat(i,i) = g2(i);
    if(mfcn.numOfCalls()  > maxcalls) {
      //std::cout<<"maxcalls " << maxcalls << " " << mfcn.numOfCalls() << "  " <<   st.nfcn() << std::endl;
      std::cout<<"MnHesse: maximum number of allowed function calls exhausted."<<std::endl;  
      std::cout<<"MnHesse fails and will return diagonal matrix "<<std::endl;
      for(unsigned int j = 0; j < n; j++) {
	double tmp = g2(j) < prec.eps2() ? 1. : 1./g2(j);
	vhmat(j,j) = tmp < prec.eps2() ? 1. : tmp;
      }
      
      return MinimumState(st.parameters(), MinimumError(vhmat, MinimumError::MnHesseFailed()), st.gradient(), st.edm(), mfcn.numOfCalls());
    }
    
  }

  if(theStrategy.strategy() > 0) {
    // refine first derivative
    HessianGradientCalculator hgc(mfcn, trafo, theStrategy);
    FunctionGradient gr = hgc(st.parameters(), FunctionGradient(grd, g2, gst));
    grd = gr.grad();
  }

  //off-diagonal elements  
  for(unsigned int i = 0; i < n; i++) {
    x(i) += dirin(i);
    for(unsigned int j = i+1; j < n; j++) {
      x(j) += dirin(j);
      double fs1 = mfcn(x);
      double elem = (fs1 + amin - yy(i) - yy(j))/(dirin(i)*dirin(j));
      vhmat(i,j) = elem;
      x(j) -= dirin(j);
    }
    x(i) -= dirin(i);
  }
  
  //verify if matrix pos-def (still 2nd derivative)
  MinimumError tmp = MnPosDef()(MinimumError(vhmat,1.), prec);
  vhmat = tmp.invHessian();
  int ifail = invert(vhmat);
  if(ifail != 0) {
    std::cout<<"MnHesse: matrix inversion fails!"<<std::endl;
    std::cout<<"MnHesse fails and will return diagonal matrix."<<std::endl;

    MnAlgebraicSymMatrix tmpsym(vhmat.nrow());
    for(unsigned int j = 0; j < n; j++) {
      double tmp = g2(j) < prec.eps2() ? 1. : 1./g2(j);
      tmpsym(j,j) = tmp < prec.eps2() ? 1. : tmp;
    }

    return MinimumState(st.parameters(), MinimumError(tmpsym, MinimumError::MnHesseFailed()), st.gradient(), st.edm(), mfcn.numOfCalls());
  }
  
  FunctionGradient gr(grd, g2, gst);

  // needed this ? (if posdef and inversion ok continue. it is like this in the Fortran version
//   if(tmp.isMadePosDef()) {
//     std::cout<<"MnHesse: matrix is invalid!"<<std::endl;
//     std::cout<<"MnHesse: matrix is not pos. def.!"<<std::endl;
//     std::cout<<"MnHesse: matrix was forced pos. def."<<std::endl;
//     return MinimumState(st.parameters(), MinimumError(vhmat, MinimumError::MnMadePosDef()), gr, st.edm(), mfcn.numOfCalls());    
//   }

  //calculate edm
  MinimumError err(vhmat, 0.);
  VariableMetricEDMEstimator estim;
  double edm = estim.estimate(gr, err);

  return MinimumState(st.parameters(), err, gr, edm, mfcn.numOfCalls());
}

/*
MinimumError MnHesse::hessian(const MnFcn& mfcn, const MinimumState& st, const MnUserTransformation& trafo) const {
  
  const MnMachinePrecision& prec = trafo.precision();
  // make sure starting at the right place
  double amin = mfcn(st.vec());
//   if(fabs(amin - st.fval()) > prec.eps2()) std::cout<<"function value differs from amin  by "<<amin - st.fval()<<std::endl;

  double aimsag = sqrt(prec.eps2())*(fabs(amin)+mfcn.up());
  
  // diagonal elements first

  unsigned int n = st.parameters().vec().size();
  MnAlgebraicSymMatrix vhmat(n);
  MnAlgebraicVector g2 = st.gradient().g2();
  MnAlgebraicVector gst = st.gradient().gstep();
  MnAlgebraicVector grd = st.gradient().grad();
  MnAlgebraicVector dirin = st.gradient().gstep();
  MnAlgebraicVector yy(n);
  MnAlgebraicVector x = st.parameters().vec(); 

  for(unsigned int i = 0; i < n; i++) {

    double xtf = x(i);
    double dmin = 8.*prec.eps2()*fabs(xtf);
    double d = fabs(gst(i));
    if(d < dmin) d = dmin;
    for(int icyc = 0; icyc < ncycles(); icyc++) {
      double sag = 0.;
      double fs1 = 0.;
      double fs2 = 0.;
      for(int multpy = 0; multpy < 5; multpy++) {
	x(i) = xtf + d;
	fs1 = mfcn(x);
	x(i) = xtf - d;
	fs2 = mfcn(x);
	x(i) = xtf;
	sag = 0.5*(fs1+fs2-2.*amin);
	if(sag > prec.eps2()) break;
	if(trafo.parameter(i).hasLimits()) {
	  if(d > 0.5) {
	    std::cout<<"second derivative zero for parameter "<<i<<std::endl;
	    std::cout<<"return diagonal matrix "<<std::endl;
	    for(unsigned int j = 0; j < n; j++) {
	      vhmat(j,j) = (g2(j) < prec.eps2() ? 1. : 1./g2(j));
 	      return MinimumError(vhmat, 1., false);
	    }
	  }
	  d *= 10.;
	  if(d > 0.5) d = 0.51;
	  continue;
	}
	d *= 10.;
      }
      if(sag < prec.eps2()) {
	std::cout<<"MnHesse: internal loop exhausted, return diagonal matrix."<<std::endl;
	for(unsigned int i = 0; i < n; i++)
	  vhmat(i,i) = (g2(i) < prec.eps2() ? 1. : 1./g2(i));
	return MinimumError(vhmat, 1., false);
      }
      double g2bfor = g2(i);
      g2(i) = 2.*sag/(d*d);
      grd(i) = (fs1-fs2)/(2.*d);
      gst(i) = d;
      dirin(i) = d;
      yy(i) = fs1;
      double dlast = d;
      d = sqrt(2.*aimsag/fabs(g2(i)));
      if(trafo.parameter(i).hasLimits()) d = std::min(0.5, d);
      if(d < dmin) d = dmin;

      // see if converged
      if(fabs((d-dlast)/d) < tolerstp()) break;
      if(fabs((g2(i)-g2bfor)/g2(i)) < tolerg2()) break; 
      d = std::min(d, 10.*dlast);
      d = std::max(d, 0.1*dlast);
    }
    vhmat(i,i) = g2(i);
  }

  //off-diagonal elements  
  for(unsigned int i = 0; i < n; i++) {
    x(i) += dirin(i);
    for(unsigned int j = i+1; j < n; j++) {
      x(j) += dirin(j);
      double fs1 = mfcn(x);
      double elem = (fs1 + amin - yy(i) - yy(j))/(dirin(i)*dirin(j));
      vhmat(i,j) = elem;
      x(j) -= dirin(j);
    }
    x(i) -= dirin(i);
  }
  
  return MinimumError(vhmat, 0.);
}
*/
