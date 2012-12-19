#ifndef MN_MnUserParameterState_H_
#define MN_MnUserParameterState_H_

#include "Minuit/MnUserParameters.h"
#include "Minuit/MnUserCovariance.h"
#include "Minuit/MnGlobalCorrelationCoeff.h"

class MinimumState;

/** class which holds the external user and/or internal Minuit representation
    of the parameters and errors;
    transformation internal <-> external on demand;
 */

class MnUserParameterState {

public:

  /// default constructor (invalid state)
  MnUserParameterState() : theValid(false), theCovarianceValid(false), theParameters(MnUserParameters()), theCovariance(MnUserCovariance()), theIntParameters(std::vector<double>()), theIntCovariance(MnUserCovariance()) {} 

  /// construct from user parameters (before minimization)
  MnUserParameterState(const std::vector<double>&, const std::vector<double>&);

  MnUserParameterState(const MnUserParameters&);

  /// construct from user parameters + covariance (before minimization)
  MnUserParameterState(const std::vector<double>&, const std::vector<double>&, unsigned int);

  MnUserParameterState(const std::vector<double>&, const MnUserCovariance&);

  MnUserParameterState(const MnUserParameters&, const MnUserCovariance&);

  /// construct from internal parameters (after minimization)
  MnUserParameterState(const MinimumState&, double, const MnUserTransformation&);

  ~MnUserParameterState() {}

  MnUserParameterState(const MnUserParameterState& state) : theValid(state.theValid), theCovarianceValid(state.theCovarianceValid), theGCCValid(state.theGCCValid), theFVal(state.theFVal), theEDM(state.theEDM), theNFcn(state.theNFcn), theParameters(state.theParameters), theCovariance(state.theCovariance), theGlobalCC(state.theGlobalCC), theIntParameters(state.theIntParameters), theIntCovariance(state.theIntCovariance) {}

  MnUserParameterState& operator=(const MnUserParameterState& state) {
    theValid = state.theValid;
    theCovarianceValid = state.theCovarianceValid;
    theGCCValid = state.theGCCValid;
    theFVal = state.theFVal;
    theEDM = state.theEDM;
    theNFcn = state.theNFcn;
    theParameters = state.theParameters;
    theCovariance = state.theCovariance;
    theGlobalCC = state.theGlobalCC;
    theIntParameters = state.theIntParameters;
    theIntCovariance = state.theIntCovariance;
    return *this;
  }

  //user external representation
  const MnUserParameters& parameters() const {return theParameters;}
  const MnUserCovariance& covariance() const {return theCovariance;}
  const MnGlobalCorrelationCoeff& globalCC() const {return theGlobalCC;}

  //Minuit internal representation
  const std::vector<double>& intParameters() const {return theIntParameters;}
  const MnUserCovariance& intCovariance() const {return theIntCovariance;}

  //transformation internal <-> external
  const MnUserTransformation& trafo() const {return theParameters.trafo();}

  bool isValid() const {return theValid;}
  bool hasCovariance() const {return theCovarianceValid;}
  bool hasGlobalCC() const {return theGCCValid;}

  double fval() const {return theFVal;}
  double edm() const {return theEDM;}
  unsigned int nfcn() const {return theNFcn;}

private:

  bool theValid;
  bool theCovarianceValid;
  bool theGCCValid;

  double theFVal;
  double theEDM;
  unsigned int theNFcn;

  MnUserParameters theParameters;
  MnUserCovariance theCovariance;
  MnGlobalCorrelationCoeff theGlobalCC;

  std::vector<double> theIntParameters;
  MnUserCovariance theIntCovariance;

public:

// facade: forward interface of MnUserParameters and MnUserTransformation

  //access to parameters (row-wise)
  const std::vector<MinuitParameter>& minuitParameters() const;
  //access to parameters and errors in column-wise representation
  std::vector<double> params() const;
  std::vector<double> errors() const;

  //access to single parameter
  const MinuitParameter& parameter(unsigned int i) const;

  //add free parameter
  void add(const char* name, double val, double err);
  //add limited parameter
  void add(const char* name, double val, double err, double , double);
  //add const parameter
  void add(const char*, double);

  //interaction via external number of parameter
  void fix(unsigned int);
  void release(unsigned int);
  void setValue(unsigned int, double);
  void setError(unsigned int, double);
  void setLimits(unsigned int, double, double);
  void setUpperLimit(unsigned int, double);
  void setLowerLimit(unsigned int, double);
  void removeLimits(unsigned int);

  double value(unsigned int) const;
  double error(unsigned int) const;
  
  //interaction via name of parameter
  void fix(const char*);
  void release(const char*);
  void setValue(const char*, double);
  void setError(const char*, double);
  void setLimits(const char*, double, double);
  void setUpperLimit(const char*, double);
  void setLowerLimit(const char*, double);
  void removeLimits(const char*);

  double value(const char*) const;
  double error(const char*) const;
  
  //convert name into external number of parameter
  unsigned int index(const char*) const;
  //convert external number into name of parameter
  const char* name(unsigned int) const;

  // transformation internal <-> external
  double int2ext(unsigned int, double) const;
  double ext2int(unsigned int, double) const;
  unsigned int intOfExt(unsigned int) const;
  unsigned int extOfInt(unsigned int) const;
  unsigned int variableParameters() const;
  const MnMachinePrecision& precision() const;
  void setPrecision(double eps);
};

#endif //MN_MnUserParameterState_H_
