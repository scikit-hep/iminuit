#ifndef MN_MnMachinePrecision_H_
#define MN_MnMachinePrecision_H_

#include <math.h>

/** 
    determines the relative floating point arithmetic precision. The 
    setPrecision() method can be used to override Minuit's own determination, 
    when the user knows that the {FCN} function value is not calculated to 
    the nominal machine accuracy.
 */

class MnMachinePrecision {

public:

  MnMachinePrecision();

  ~MnMachinePrecision() {}

  MnMachinePrecision(const MnMachinePrecision& prec) : theEpsMac(prec.theEpsMac), theEpsMa2(prec.theEpsMa2) {}

  MnMachinePrecision& operator=(const MnMachinePrecision& prec) {
    theEpsMac = prec.theEpsMac;
    theEpsMa2 = prec.theEpsMa2;
    return *this;
  }

  /// eps returns the smallest possible number so that 1.+eps > 1.
  double eps() const {return theEpsMac;}

  /// eps2 returns 2*sqrt(eps)
  double eps2() const {return theEpsMa2;}

  /// override Minuit's own determination
  void setPrecision(double prec) {
    theEpsMac = prec;
    theEpsMa2 = 2.*sqrt(theEpsMac);
  }

private:

  double theEpsMac;
  double theEpsMa2;
};

#endif // MN_MnMachinePrecision_H_
