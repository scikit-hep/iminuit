#ifndef MN_MnUserTransformation_H_
#define MN_MnUserTransformation_H_

#include "Minuit/MnConfig.h"
#include "Minuit/MnMatrix.h"
#include "Minuit/MinuitParameter.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/SinParameterTransformation.h"
#include "Minuit/SqrtLowParameterTransformation.h"
#include "Minuit/SqrtUpParameterTransformation.h"

#include <vector>

class MnUserCovariance;

// class MnMachinePrecision;

/** knows how to transform between user specified parameters (external) and
    internal parameters used for minimization
 */

class MnUserTransformation {

public:

  MnUserTransformation() : thePrecision(MnMachinePrecision()),
			   theParameters(std::vector<MinuitParameter>()),
			   theExtOfInt(std::vector<unsigned int>()),
			   theDoubleLimTrafo(SinParameterTransformation()),
			   theUpperLimTrafo(SqrtUpParameterTransformation()),
			   theLowerLimTrafo(SqrtLowParameterTransformation()),
			   theCache(std::vector<double>()) {}

  MnUserTransformation(const std::vector<double>&, const std::vector<double>&);

  ~MnUserTransformation() {}

  MnUserTransformation(const MnUserTransformation& trafo) : 
    thePrecision(trafo.thePrecision),
    theParameters(trafo.theParameters),theExtOfInt(trafo.theExtOfInt), 
    theDoubleLimTrafo(trafo.theDoubleLimTrafo), 
    theUpperLimTrafo(trafo.theUpperLimTrafo), 
    theLowerLimTrafo(trafo.theLowerLimTrafo), theCache(trafo.theCache) {}
  
  MnUserTransformation& operator=(const MnUserTransformation& trafo) {
    thePrecision = trafo.thePrecision;
    theParameters = trafo.theParameters;
    theExtOfInt = trafo.theExtOfInt;
    theDoubleLimTrafo = trafo.theDoubleLimTrafo;
    theUpperLimTrafo = trafo.theUpperLimTrafo;
    theLowerLimTrafo = trafo.theLowerLimTrafo;
    theCache = trafo.theCache;
    return *this;
  }

  const std::vector<double>& operator()(const MnAlgebraicVector&) const;

  // index = internal parameter
  double int2ext(unsigned int, double) const;

  // index = internal parameter
  double int2extError(unsigned int, double, double) const;

  MnUserCovariance int2extCovariance(const MnAlgebraicVector&, const MnAlgebraicSymMatrix&) const;

  // index = external parameter
  double ext2int(unsigned int, double) const;

  // index = internal parameter
  double dInt2Ext(unsigned int, double) const;

//   // index = external parameter
//   double dExt2Int(unsigned int, double) const;

  // index = external parameter
  unsigned int intOfExt(unsigned int) const;

  // index = internal parameter
  unsigned int extOfInt(unsigned int internal) const { 
    assert(internal < theExtOfInt.size());
    return theExtOfInt[internal];
  }

  const std::vector<MinuitParameter>& parameters() const {
    return theParameters;
  }

  unsigned int variableParameters() const {return static_cast<unsigned int> ( theExtOfInt.size() );}

private:

  MnMachinePrecision thePrecision;
  std::vector<MinuitParameter> theParameters;
  std::vector<unsigned int> theExtOfInt;

  SinParameterTransformation theDoubleLimTrafo;
  SqrtUpParameterTransformation theUpperLimTrafo;
  SqrtLowParameterTransformation theLowerLimTrafo;

  mutable std::vector<double> theCache;

public:

  //forwarded interface
  const MnMachinePrecision& precision() const {return thePrecision;}
  void setPrecision(double eps) {thePrecision.setPrecision(eps);}

  //access to parameters and errors in column-wise representation 
  std::vector<double> params() const;
  std::vector<double> errors() const;

  //access to single parameter
  const MinuitParameter& parameter(unsigned int) const;

  //add free parameter
  bool add(const char*, double, double);
  //add limited parameter
  bool add(const char*, double, double, double, double);
  //add const parameter
  bool add(const char*, double);

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

};

#endif //MN_MnUserTransformation_H_
