#ifndef MN_MinuitParameter_H_
#define MN_MinuitParameter_H_

#include <algorithm>
#include <memory>
#include <string>
#include <iostream>

/** class for the individual Minuit parameter with name and number;
    contains the input numbers for the minimization or the output result
    from minimization;
    possible interactions: fix/release, set/remove limits, set value/error;
 */

class MinuitParameter {

public:

  //constructor for constant parameter
  MinuitParameter(unsigned int num, const char* name, double val) :
    theNum(num), theName(name), theValue(val), theError(0.), theConst(true), theFix(false),
    theLoLimit(0.), theUpLimit(0.), theLoLimValid(false), theUpLimValid(false){
    }

  //constructor for standard parameter
  MinuitParameter(unsigned int num, const char* name, double val, double err) :
    theNum(num), theName(name), theValue(val), theError(err), theConst(false), theFix(false),
    theLoLimit(0.), theUpLimit(0.), theLoLimValid(false), theUpLimValid(false){
    }

  //constructor for limited parameter
  MinuitParameter(unsigned int num, const char* name, double val, double err,
		  double min, double max) :
    theNum(num), theName(name),theValue(val), theError(err), theConst(false),
    theFix(false), theLoLimit(min), theUpLimit(max), theLoLimValid(true),
    theUpLimValid(true){
    assert(min != max);
    if(min > max) {
      theLoLimit = max;
      theUpLimit = min;
    }
  }

  ~MinuitParameter() {}

  MinuitParameter(const MinuitParameter& par) :
    theNum(par.theNum), theName(par.theName), theValue(par.theValue), theError(par.theError),
    theConst(par.theConst), theFix(par.theFix), theLoLimit(par.theLoLimit),
    theUpLimit(par.theUpLimit), theLoLimValid(par.theLoLimValid),
    theUpLimValid(par.theUpLimValid) {

  }

  MinuitParameter& operator=(const MinuitParameter& par) {
    theNum = par.theNum;
    theName = par.theName;
    theValue = par.theValue;
    theError = par.theError;
    theConst = par.theConst;
    theFix = par.theFix;
    theLoLimit = par.theLoLimit;
    theUpLimit = par.theUpLimit;
    theLoLimValid = par.theLoLimValid;
    theUpLimValid = par.theUpLimValid;
    return *this;
  }

  //access methods
  unsigned int number() const {return theNum;}
  const char* name() const {return theName.c_str();}
  double value() const {return theValue;}
  double error() const {return theError;}

  //interaction
  void setValue(double val) {theValue = val;}
  void setError(double err) {theError = err;}
  void setLimits(double low, double up) {
    assert(low != up);
    theLoLimit = low;
    theUpLimit = up;
    theLoLimValid = true;
    theUpLimValid = true;
    if(low > up) {
      theLoLimit = up;
      theUpLimit = low;
    }
  }

  void setUpperLimit(double up) {
    theLoLimit = 0.;
    theUpLimit = up;
    theLoLimValid = false;
    theUpLimValid = true;
  }

  void setLowerLimit(double low) {
    theLoLimit = low;
    theUpLimit = 0.;
    theLoLimValid = true;
    theUpLimValid = false;
  }

  void removeLimits() {
    theLoLimit = 0.;
    theUpLimit = 0.;
    theLoLimValid = false;
    theUpLimValid = false;
  }

  void fix() {theFix = true;}
  void release() {theFix = false;}

  //state of parameter (fixed/const/limited)
  bool isConst() const {return theConst;}
  bool isFixed() const {return theFix;}

  bool hasLimits() const {return theLoLimValid || theUpLimValid; }
  bool hasLowerLimit() const {return theLoLimValid; }
  bool hasUpperLimit() const {return theUpLimValid; }
  double lowerLimit() const {return theLoLimit;}
  double upperLimit() const {return theUpLimit;}

private:

  unsigned int theNum;
  std::string theName;
  double theValue;
  double theError;
  bool theConst;
  bool theFix;
  double theLoLimit;
  double theUpLimit;
  bool theLoLimValid;
  bool theUpLimValid;

};

#endif //MN_MinuitParameter_H_
