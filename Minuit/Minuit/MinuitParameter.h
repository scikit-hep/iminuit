#ifndef MN_MinuitParameter_H_
#define MN_MinuitParameter_H_

#include <algorithm>
#include <memory>

/** class for the individual Minuit parameter with name and number;
    contains the input numbers for the minimization or the output result
    from minimization;
    possible interactions: fix/release, set/remove limits, set value/error;
 */

class MinuitParameter {

public:

  //constructor for constant parameter
  MinuitParameter(unsigned int num, const char* name, double val) :
    theNum(num), theValue(val), theError(0.), theConst(true), theFix(false),
    theLoLimit(0.), theUpLimit(0.), theLoLimValid(false), theUpLimValid(false){
    setName(name);
  }

  //constructor for standard parameter
  MinuitParameter(unsigned int num, const char* name, double val, double err) :
    theNum(num), theValue(val), theError(err), theConst(false), theFix(false),
    theLoLimit(0.), theUpLimit(0.), theLoLimValid(false), theUpLimValid(false){
    setName(name);
  }

  //constructor for limited parameter
  MinuitParameter(unsigned int num, const char* name, double val, double err,
		  double min, double max) :
    theNum(num),theValue(val), theError(err), theConst(false), theFix(false),
    theLoLimit(min), theUpLimit(max), theLoLimValid(true), theUpLimValid(true){
    assert(min != max);
    if(min > max) {
      theLoLimit = max;
      theUpLimit = min;
    }
    setName(name);
  }

  ~MinuitParameter() {}

  MinuitParameter(const MinuitParameter& par) :
    theNum(par.theNum), theValue(par.theValue), theError(par.theError),
    theConst(par.theConst), theFix(par.theFix), theLoLimit(par.theLoLimit),
    theUpLimit(par.theUpLimit), theLoLimValid(par.theLoLimValid),
    theUpLimValid(par.theUpLimValid) {
    memcpy(theName, par.name(), 11*sizeof(char));
  }

  MinuitParameter& operator=(const MinuitParameter& par) {
    theNum = par.theNum;
    memcpy(theName, par.theName, 11*sizeof(char));
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
  const char* name() const {return theName;}
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
  char theName[11];
  double theValue;
  double theError;
  bool theConst;
  bool theFix;
  double theLoLimit;
  double theUpLimit;
  bool theLoLimValid;
  bool theUpLimValid;

private:

  void setName(const char* name) {
    int l = std::min(int(strlen(name)), 11);
    memset(theName, 0, 11*sizeof(char));
    memcpy(theName, name, l*sizeof(char));
    theName[10] = '\0';
  }
};

#endif //MN_MinuitParameter_H_
