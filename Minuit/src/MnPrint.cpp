#include "Minuit/MnPrint.h"
#include "Minuit/LAVector.h"
#include "Minuit/LASymMatrix.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnUserParameters.h"
#include "Minuit/MnUserCovariance.h"
#include "Minuit/MnGlobalCorrelationCoeff.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MinuitParameter.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MinosError.h"
#include "Minuit/ContoursError.h"
#include "Minuit/MnPlot.h"

#include <iomanip>

std::ostream& operator<<(std::ostream& os, const LAVector& vec) {
  os << "LAVector parameters:" << std::endl;
  { 
    os << std::endl;
    int nrow = vec.size();
    for (int i = 0; i < nrow; i++) {
      os.precision(6); os.width(13); 
      os << vec(i) << std::endl;
    }
    os << std::endl;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const LASymMatrix& matrix) {
  os << "LASymMatrix parameters:" << std::endl;
   { 
    os << std::endl;
    int n = matrix.nrow();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        os.precision(6); os.width(13); os << matrix(i,j);
      }
      os << std::endl;
    }
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const MnUserParameters& par) {

  os << std::endl;
 
  os << "# ext. |" << "|   name    |" << "|   type  |" << "|   value   |" << "|  error +/- " << std::endl;
      
  os << std::endl;

  bool atLoLim = false;
  bool atHiLim = false;
  for(std::vector<MinuitParameter>::const_iterator ipar = par.parameters().begin(); ipar != par.parameters().end(); ipar++) {
    os << std::setw(4) << (*ipar).number() << std::setw(5) << "||"; 
    os << std::setw(10) << (*ipar).name()   << std::setw(3) << "||";
    if((*ipar).isConst()) {
      os << "  const  ||" << std::setprecision(4) << std::setw(10) << (*ipar).value() << " ||" << std::endl;
    } else if((*ipar).isFixed()) {
      os << "  fixed  ||" << std::setprecision(4) << std::setw(10) << (*ipar).value() << " ||" << std::endl;
    } else if((*ipar).hasLimits()) {
      if((*ipar).error() > 0.) {
	os << " limited ||" << std::setprecision(4) << std::setw(10) << (*ipar).value();
	if(fabs((*ipar).value() - (*ipar).lowerLimit()) < par.precision().eps2()) {
	  os <<"*";
	  atLoLim = true;
	}
	if(fabs((*ipar).value() - (*ipar).upperLimit()) < par.precision().eps2()) {
	  os <<"**";
	  atHiLim = true;
	}
	os << " ||" << std::setw(8) << (*ipar).error() << std::endl;
      } else
	os << "  free   ||" << std::setprecision(4) << std::setw(10) << (*ipar).value() << " ||" << std::setw(8) << "no" << std::endl;
    } else {
      if((*ipar).error() > 0.)
	os << "  free   ||" << std::setprecision(4) << std::setw(10) << (*ipar).value() << " ||" << std::setw(8) << (*ipar).error() << std::endl;
      else
	os << "  free   ||" << std::setprecision(4) << std::setw(10) << (*ipar).value() << " ||" << std::setw(8) << "no" << std::endl;
	
    }
  }
  os << std::endl;
  if(atLoLim) os << "* parameter is at lower limit" << std::endl;
  if(atHiLim) os << "** parameter is at upper limit" << std::endl;
  os << std::endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const MnUserCovariance& matrix) {

  os << std::endl;

  os << "MnUserCovariance: " << std::endl;

  { 
    os << std::endl;
    unsigned int n = matrix.nrow();
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int j = 0; j < n; j++) {
        os.precision(6); os.width(13); os << matrix(i,j);
      }
      os << std::endl;
    }
  }

   os << std::endl;
   os << "MnUserCovariance parameter correlations: " << std::endl;

   { 
     os << std::endl;
    unsigned int n = matrix.nrow();
    for (unsigned int i = 0; i < n; i++) {
      double di = matrix(i,i);
      for (unsigned int j = 0; j < n; j++) {
	double dj = matrix(j,j);	
        os.precision(6); os.width(13); os << matrix(i,j)/sqrt(fabs(di*dj));
      }
      os << std::endl;
    }
  }

  return os;   
}

std::ostream& operator<<(std::ostream& os, const MnGlobalCorrelationCoeff& coeff) {

  os << std::endl;

  os << "MnGlobalCorrelationCoeff: " << std::endl;

   { 
    os << std::endl;
    for (unsigned int i = 0; i < coeff.globalCC().size(); i++) {
      os.precision(6); os.width(13); os << coeff.globalCC()[i];
      os << std::endl;
    }
  }

  return os;   
}

std::ostream& operator<<(std::ostream& os, const MnUserParameterState& state) {

  os << std::endl;

  if(!state.isValid()) {
    os << std::endl;
    os <<"WARNING: MnUserParameterState is not valid."<<std::endl;
    os << std::endl;
  }
    
  os <<"# of function calls: "<<state.nfcn()<<std::endl;
  os <<"function value: "<<state.fval()<<std::endl;
  os <<"expected distance to the minimum (edm): "<<state.edm()<<std::endl;
  os <<"external parameters: "<<state.parameters()<<std::endl;
  if(state.hasCovariance())
    os <<"covariance matrix: "<<state.covariance()<<std::endl;
  if(state.hasGlobalCC()) 
    os <<"global correlation coefficients : "<<state.globalCC()<<std::endl;
  
  if(!state.isValid())
    os <<"WARNING: MnUserParameterState is not valid."<<std::endl;

  os << std::endl;

  return os;
} 

std::ostream& operator<<(std::ostream& os, const FunctionMinimum& min) {

  os << std::endl;
  if(!min.isValid()) {
    os << std::endl;
    os <<"WARNING: Minuit did not converge."<<std::endl;
    os << std::endl;
  } else {
    os << std::endl;
    os <<"Minuit did successfully converge."<<std::endl;
    os << std::endl;
  }
    
  os <<"# of function calls: "<<min.nfcn()<<std::endl;
  os <<"minimum function value: "<<min.fval()<<std::endl;
  os <<"minimum edm: "<<min.edm()<<std::endl;
  os <<"minimum internal state vector: "<<min.parameters().vec()<<std::endl;
  if(min.hasValidCovariance()) 
    os <<"minimum internal covariance matrix: "<<min.error().matrix()<<std::endl;
  
  os << min.userParameters() << std::endl;
  os << min.userCovariance() << std::endl;
  os << min.userState().globalCC() << std::endl;

  if(!min.isValid())
    os <<"WARNING: FunctionMinimum is invalid."<<std::endl;

  os << std::endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const MinimumState& min) {

  os << std::endl;

  os <<"minimum function value: "<<min.fval()<<std::endl;
  os <<"minimum edm: "<<min.edm()<<std::endl;
  os <<"minimum internal state vector: "<<min.vec()<<std::endl;
  os <<"minimum internal gradient vector: "<<min.gradient().vec()<<std::endl;
  if(min.hasCovariance()) 
    os <<"minimum internal covariance matrix: "<<min.error().matrix()<<std::endl;
 
  os << std::endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const MnMachinePrecision& prec) {

  os << std::endl;
  
  os <<"current machine precision is set to "<<prec.eps()<<std::endl;

  os << std::endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const MinosError& me) {

  os << std::endl;

  os <<"Minos # of function calls: "<<me.nfcn()<<std::endl;
  
  if(!me.isValid())
    os << "Minos error is not valid." <<std::endl;
  if(!me.lowerValid())
    os << "lower Minos error is not valid." <<std::endl;
  if(!me.upperValid())
    os << "upper Minos error is not valid." <<std::endl;
  if(me.atLowerLimit())
    os << "Minos error is lower limit of parameter "<<me.parameter()<<"." <<std::endl;
  if(me.atUpperLimit())
    os << "Minos error is upper limit of parameter "<<me.parameter()<<"." <<std::endl;
  if(me.atLowerMaxFcn())
    os << "Minos number of function calls for lower error exhausted."<<std::endl;
  if(me.atUpperMaxFcn())
    os << "Minos number of function calls for upper error exhausted."<<std::endl;
  if(me.lowerNewMin()) {
    os << "Minos found a new minimum in negative direction."<<std::endl;
    os << me.lowerState() <<std::endl;
  }
  if(me.upperNewMin()) {
    os << "Minos found a new minimum in positive direction."<<std::endl;
    os << me.upperState() <<std::endl;
  }

  os << "# ext. |" << "|   name    |" << "| value@min |" << "|  negative |" << "| positive  " << std::endl;
  os << std::setw(4) << me.parameter() << std::setw(5) << "||"; 
  os << std::setw(10) << me.lowerState().name(me.parameter()) << std::setw(3) << "||";
  os << std::setprecision(4) << std::setw(10) << me.min() << " ||" << std::setprecision(4) << std::setw(10) << me.lower() << " ||" << std::setw(8) << me.upper() << std::endl;
  
  os << std::endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const ContoursError& ce) {

  os << std::endl;
  os <<"Contours # of function calls: "<<ce.nfcn()<<std::endl;
  os << "MinosError in x: "<<std::endl;
  os << ce.xMinosError() << std::endl;
  os << "MinosError in y: "<<std::endl;
  os << ce.yMinosError() << std::endl;
  MnPlot plot;
  plot(ce.xmin(), ce.ymin(), ce());
  for(std::vector<std::pair<double,double> >::const_iterator ipar = ce().begin(); ipar != ce().end(); ipar++) {
    os << ipar - ce().begin() <<"  "<< (*ipar).first <<"  "<< (*ipar).second <<std::endl;
  }
  os << std::endl;

  return os;
}
