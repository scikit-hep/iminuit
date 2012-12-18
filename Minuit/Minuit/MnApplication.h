#ifndef MN_MnApplication_H_
#define MN_MnApplication_H_

#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnStrategy.h"


class FunctionMinimum;
class MinuitParameter;
class MnMachinePrecision;
class ModularFunctionMinimizer;
class FCNBase;


/** application interface class for minimizers (migrad, simplex, minimize, 
    scan)
 */

class MnApplication {

public:

  MnApplication(const FCNBase& fcn, const MnUserParameterState& state, const MnStrategy& stra) : theFCN(fcn), theState(state), theStrategy(stra), theNumCall(0) {}

  MnApplication(const FCNBase& fcn, const MnUserParameterState& state, const MnStrategy& stra, unsigned int nfcn) : theFCN(fcn), theState(state), theStrategy(stra), theNumCall(nfcn) {}

  virtual ~MnApplication() { }

  /// minimize
  virtual FunctionMinimum operator()(unsigned int = 0, double = 0.1);
 
  virtual const ModularFunctionMinimizer& minimizer() const = 0;

  const MnMachinePrecision& precision() const {return theState.precision();}
  const MnUserParameterState& state() const {return theState;}
  const MnUserParameters& parameters() const {return theState.parameters();}
  const MnUserCovariance& covariance() const {return theState.covariance();}
  virtual const FCNBase& fcnbase() const {return theFCN;}
  const MnStrategy& strategy() const {return theStrategy;}
  unsigned int numOfCalls() const {return theNumCall;}

protected:

  const FCNBase& theFCN;
  MnUserParameterState theState;
  MnStrategy theStrategy;
  unsigned int theNumCall;

public:

// facade: forward interface of MnUserParameters and MnUserTransformation
// via MnUserParameterState

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
  void removeLimits(unsigned int);

  double value(unsigned int) const;
  double error(unsigned int) const;
  
  //interaction via name of parameter
  void fix(const char*);
  void release(const char*);
  void setValue(const char*, double);
  void setError(const char*, double);
  void setLimits(const char*, double, double);
  void removeLimits(const char*);
  void setPrecision(double);

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

};

#endif //MN_MnApplication_H_
