#include "Minuit/MnUserTransformation.h"
#include "Minuit/MnUserCovariance.h"

#include <algorithm>
#include <cstdio>

class MnParStr {

public:

  MnParStr(const char* name) : theName(name) {}

  ~MnParStr() {}
  
  bool operator()(const MinuitParameter& par) const {
    return (strcmp(par.name(), theName) == 0);
  }

private:
  const char* theName;
};

MnUserTransformation::MnUserTransformation(const std::vector<double>& par, const std::vector<double>& err) : thePrecision(MnMachinePrecision()), theParameters(std::vector<MinuitParameter>()), theExtOfInt(std::vector<unsigned int>()), theDoubleLimTrafo(SinParameterTransformation()),theUpperLimTrafo(SqrtUpParameterTransformation()), theLowerLimTrafo(SqrtLowParameterTransformation()), theCache(std::vector<double>()) {
  theParameters.reserve(par.size());
  theExtOfInt.reserve(par.size());
  theCache.reserve(par.size());
  char p[5];
  p[0] = 'p';
  p[4] = '\0';
  for(unsigned int i = 0; i < par.size(); i++) {
    std::sprintf(p+1,"%i",i);
    add(p, par[i], err[i]);
  }
}

const std::vector<double>& MnUserTransformation::operator()(const MnAlgebraicVector& pstates) const {

  for(unsigned int i = 0; i < pstates.size(); i++) {
    if(theParameters[theExtOfInt[i]].hasLimits()) {
      theCache[theExtOfInt[i]] = int2ext(i, pstates(i));
    } else {
      theCache[theExtOfInt[i]] = pstates(i);
    }
  }

  return theCache;
}

double MnUserTransformation::int2ext(unsigned int i, double val) const {

  if(theParameters[theExtOfInt[i]].hasLimits()) {
    if(theParameters[theExtOfInt[i]].hasUpperLimit() && theParameters[theExtOfInt[i]].hasLowerLimit())
      return theDoubleLimTrafo.int2ext(val, theParameters[theExtOfInt[i]].upperLimit(), theParameters[theExtOfInt[i]].lowerLimit());
    else if(theParameters[theExtOfInt[i]].hasUpperLimit() && !theParameters[theExtOfInt[i]].hasLowerLimit())
      return theUpperLimTrafo.int2ext(val, theParameters[theExtOfInt[i]].upperLimit());
    else
      return theLowerLimTrafo.int2ext(val, theParameters[theExtOfInt[i]].lowerLimit());
  }

  return val;
}

double MnUserTransformation::int2extError(unsigned int i, double val, double err) const {
  //err = sigma value == sqrt(cov(i,i))
  double dx = err;
  
  if(theParameters[theExtOfInt[i]].hasLimits()) {
    double ui = int2ext(i, val);
    double du1 = int2ext(i, val+dx) - ui;
    double du2 = int2ext(i, val-dx) - ui;
    if(theParameters[theExtOfInt[i]].hasUpperLimit() && theParameters[theExtOfInt[i]].hasLowerLimit()) {
//       double al = theParameters[theExtOfInt[i]].lower();
//       double ba = theParameters[theExtOfInt[i]].upper() - al;
//       double du1 = al + 0.5*(sin(val + dx) + 1.)*ba - ui;
//       double du2 = al + 0.5*(sin(val - dx) + 1.)*ba - ui;
//       if(dx > 1.) du1 = ba;
      if(dx > 1.) du1 = theParameters[theExtOfInt[i]].upperLimit() - theParameters[theExtOfInt[i]].lowerLimit();
      dx = 0.5*(fabs(du1) + fabs(du2));
    } else {
      dx = 0.5*(fabs(du1) + fabs(du2));
    }
  }

  return dx;
}

MnUserCovariance MnUserTransformation::int2extCovariance(const MnAlgebraicVector& vec, const MnAlgebraicSymMatrix& cov) const {
  
  MnUserCovariance result(cov.nrow());
  for(unsigned int i = 0; i < vec.size(); i++) {
    double dxdi = 1.;
    if(theParameters[theExtOfInt[i]].hasLimits()) {
//       dxdi = 0.5*fabs((theParameters[theExtOfInt[i]].upper() - theParameters[theExtOfInt[i]].lower())*cos(vec(i)));
      dxdi = dInt2Ext(i, vec(i));
    }
    for(unsigned int j = i; j < vec.size(); j++) {
      double dxdj = 1.;
      if(theParameters[theExtOfInt[j]].hasLimits()) {
// 	dxdj = 0.5*fabs((theParameters[theExtOfInt[j]].upper() - theParameters[theExtOfInt[j]].lower())*cos(vec(j)));
	dxdj = dInt2Ext(j, vec(j));
      }
      result(i,j) = dxdi*cov(i,j)*dxdj;
    }
//     double diag = int2extError(i, vec(i), sqrt(cov(i,i)));
//     result(i,i) = diag*diag;
  }
  
  return result;
}

double MnUserTransformation::ext2int(unsigned int i, double val) const {

  if(theParameters[i].hasLimits()) {
    if(theParameters[i].hasUpperLimit() && theParameters[i].hasLowerLimit())
      return theDoubleLimTrafo.ext2int(val, theParameters[i].upperLimit(), theParameters[i].lowerLimit(), precision());
    else if(theParameters[i].hasUpperLimit() && !theParameters[i].hasLowerLimit())
      return theUpperLimTrafo.ext2int(val, theParameters[i].upperLimit(), precision());
    else 
      return theLowerLimTrafo.ext2int(val, theParameters[i].lowerLimit(), precision());
  }
  
  return val;
}

double MnUserTransformation::dInt2Ext(unsigned int i, double val) const {
  double dd = 1.;
  if(theParameters[theExtOfInt[i]].hasLimits()) {
    if(theParameters[theExtOfInt[i]].hasUpperLimit() && theParameters[theExtOfInt[i]].hasLowerLimit())  
//       dd = 0.5*fabs((theParameters[theExtOfInt[i]].upper() - theParameters[theExtOfInt[i]].lower())*cos(vec(i)));
      dd = theDoubleLimTrafo.dInt2Ext(val, theParameters[theExtOfInt[i]].upperLimit(), theParameters[theExtOfInt[i]].lowerLimit());
    else if(theParameters[theExtOfInt[i]].hasUpperLimit() && !theParameters[theExtOfInt[i]].hasLowerLimit())
      dd = theUpperLimTrafo.dInt2Ext(val, theParameters[theExtOfInt[i]].upperLimit());
    else 
      dd = theLowerLimTrafo.dInt2Ext(val, theParameters[theExtOfInt[i]].lowerLimit());
  }

  return dd;
}

/*
double MnUserTransformation::dExt2Int(unsigned int, double) const {
  double dd = 1.;

  if(theParameters[theExtOfInt[i]].hasLimits()) {
    if(theParameters[theExtOfInt[i]].hasUpperLimit() && theParameters[theExtOfInt[i]].hasLowerLimit())  
//       dd = 0.5*fabs((theParameters[theExtOfInt[i]].upper() - theParameters[theExtOfInt[i]].lower())*cos(vec(i)));
      dd = theDoubleLimTrafo.dExt2Int(val, theParameters[theExtOfInt[i]].upperLimit(), theParameters[theExtOfInt[i]].lowerLimit());
    else if(theParameters[theExtOfInt[i]].hasUpperLimit() && !theParameters[theExtOfInt[i]].hasLowerLimit())
      dd = theUpperLimTrafo.dExt2Int(val, theParameters[theExtOfInt[i]].upperLimit());
    else 
      dd = theLowerLimTrafo.dExtInt(val, theParameters[theExtOfInt[i]].lowerLimit());
  }

  return dd;
}
*/

unsigned int MnUserTransformation::intOfExt(unsigned int ext) const {
  assert(ext < theParameters.size());
  assert(!theParameters[ext].isFixed());
  assert(!theParameters[ext].isConst());
  std::vector<unsigned int>::const_iterator iind = std::find(theExtOfInt.begin(), theExtOfInt.end(), ext);
  assert(iind != theExtOfInt.end());

  return (iind - theExtOfInt.begin());  
}

std::vector<double> MnUserTransformation::params() const {
  std::vector<double> result; result.reserve(theParameters.size());
  for(std::vector<MinuitParameter>::const_iterator ipar = parameters().begin();
      ipar != parameters().end(); ipar++)
    result.push_back((*ipar).value());

  return result;
}

std::vector<double> MnUserTransformation::errors() const {
  std::vector<double> result; result.reserve(theParameters.size());
  for(std::vector<MinuitParameter>::const_iterator ipar = parameters().begin();
      ipar != parameters().end(); ipar++)
    result.push_back((*ipar).error());
  
  return result;
}

const MinuitParameter& MnUserTransformation::parameter(unsigned int n) const {
  assert(n < theParameters.size()); 
  return theParameters[n];
}

bool MnUserTransformation::add(const char* name, double val, double err) {
  if (std::find_if(theParameters.begin(), theParameters.end(), MnParStr(name)) != theParameters.end() ) 
    return false; 
  theExtOfInt.push_back(theParameters.size());
  theCache.push_back(val);
  theParameters.push_back(MinuitParameter(theParameters.size(), name, val, err));
  return true;
}

bool MnUserTransformation::add(const char* name, double val, double err, double low, double up) {
  if (std::find_if(theParameters.begin(), theParameters.end(), MnParStr(name)) != theParameters.end() ) 
    return false; 
  theExtOfInt.push_back(theParameters.size());
  theCache.push_back(val);
  theParameters.push_back(MinuitParameter(theParameters.size(), name, val, err, low, up));
  return true;
}

bool MnUserTransformation::add(const char* name, double val) {
  if (std::find_if(theParameters.begin(), theParameters.end(), MnParStr(name)) != theParameters.end() ) 
    return false; 
  theCache.push_back(val);
  theParameters.push_back(MinuitParameter(theParameters.size(), name, val));
  return true;
}

void MnUserTransformation::fix(unsigned int n) {
  assert(n < theParameters.size()); 
  std::vector<unsigned int>::iterator iind = std::find(theExtOfInt.begin(), theExtOfInt.end(), n);
  assert(iind != theExtOfInt.end());
  theExtOfInt.erase(iind, iind+1);
  theParameters[n].fix();
}

void MnUserTransformation::release(unsigned int n) {
  assert(n < theParameters.size()); 
  std::vector<unsigned int>::const_iterator iind = std::find(theExtOfInt.begin(), theExtOfInt.end(), n);
  assert(iind == theExtOfInt.end());
  theExtOfInt.push_back(n);
  std::sort(theExtOfInt.begin(), theExtOfInt.end());
  theParameters[n].release();
}

void MnUserTransformation::setValue(unsigned int n, double val) {
  assert(n < theParameters.size()); 
  theParameters[n].setValue(val);
  theCache[n] = val;
}

void MnUserTransformation::setError(unsigned int n, double err) {
  assert(n < theParameters.size()); 
  theParameters[n].setError(err);
}

void MnUserTransformation::setLimits(unsigned int n, double low, double up) {
  assert(n < theParameters.size());
  assert(low != up);
  theParameters[n].setLimits(low, up);
}

void MnUserTransformation::setUpperLimit(unsigned int n, double up) {
  assert(n < theParameters.size()); 
  theParameters[n].setUpperLimit(up);
}

void MnUserTransformation::setLowerLimit(unsigned int n, double lo) {
  assert(n < theParameters.size()); 
  theParameters[n].setLowerLimit(lo);
}

void MnUserTransformation::removeLimits(unsigned int n) {
  assert(n < theParameters.size()); 
  theParameters[n].removeLimits();
}

double MnUserTransformation::value(unsigned int n) const {
  assert(n < theParameters.size()); 
  return theParameters[n].value();
}

double MnUserTransformation::error(unsigned int n) const {
  assert(n < theParameters.size()); 
  return theParameters[n].error();
}

void MnUserTransformation::fix(const char* name) {
  fix(index(name));
}

void MnUserTransformation::release(const char* name) {
  release(index(name));
}

void MnUserTransformation::setValue(const char* name, double val) {
  setValue(index(name), val);
}

void MnUserTransformation::setError(const char* name, double err) {
  setError(index(name), err);
}

void MnUserTransformation::setLimits(const char* name, double low, double up) {
  setLimits(index(name), low, up);
}

void MnUserTransformation::setUpperLimit(const char* name, double up) {
  setUpperLimit(index(name), up);
}

void MnUserTransformation::setLowerLimit(const char* name, double lo) {
  setLowerLimit(index(name), lo);
}

void MnUserTransformation::removeLimits(const char* name) {
  removeLimits(index(name));
}

double MnUserTransformation::value(const char* name) const {
  return value(index(name));
}

double MnUserTransformation::error(const char* name) const {
  return error(index(name));
}
  
unsigned int MnUserTransformation::index(const char* name) const {
  std::vector<MinuitParameter>::const_iterator ipar = 
    std::find_if(theParameters.begin(), theParameters.end(), MnParStr(name));
  assert(ipar != theParameters.end());
//   return (ipar - theParameters.begin());
  return (*ipar).number();
}

const char* MnUserTransformation::name(unsigned int n) const {
  assert(n < theParameters.size()); 
  return theParameters[n].name();
}
