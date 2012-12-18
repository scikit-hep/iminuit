#ifndef MN_MnUserParameters_H_
#define MN_MnUserParameters_H_

#include "Minuit/MnUserTransformation.h"

#include <vector>

class MnMachinePrecision;

/** API class for the user interaction with the parameters;
    serves as input to the minimizer as well as output from it;
    users can interact: fix/release parameters, set values and errors, etc.;
    parameters can be accessed via their parameter number (determined 
    internally by Minuit) or via their user-specified name (10 character 
    string); 
 */

class MnUserParameters {

public:

  MnUserParameters() : theTransformation(MnUserTransformation()) {}

  MnUserParameters(const std::vector<double>&, const std::vector<double>&);

  ~MnUserParameters() {}

  MnUserParameters(const MnUserParameters& par) :
    theTransformation(par.theTransformation) {}

  MnUserParameters& operator=(const MnUserParameters& par) {
    theTransformation = par.theTransformation;
    return *this;
  }

  const MnUserTransformation& trafo() const {return theTransformation;}

  unsigned int variableParameters() const {
    return theTransformation.variableParameters();
  }

  /// access to parameters (row-wise)
  const std::vector<MinuitParameter>& parameters() const;

  /// access to parameters and errors in column-wise representation
  std::vector<double> params() const;
  std::vector<double> errors() const;

  /// access to single parameter
  const MinuitParameter& parameter(unsigned int) const;

  /// add free parameter name, value, error
  bool add(const char*, double, double);
  /// add limited parameter name, value, lower bound, upper bound
  bool add(const char*, double, double, double, double);
  /// add const parameter name, vale
  bool add(const char*, double);

  /// interaction via external number of parameter
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

  /// interaction via name of parameter
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

  const MnMachinePrecision& precision() const;
  void setPrecision(double eps) {theTransformation.setPrecision(eps);}

private:

  MnUserTransformation theTransformation;
};

#endif //MN_MnUserParameters_H_
