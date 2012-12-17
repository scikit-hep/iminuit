#ifndef MN_MnPlot_H_
#define MN_MnPlot_H_

#include "Minuit/MnConfig.h"
#include <vector>
#include <utility>

/** MnPlot produces a text-screen graphical output of (x,y) points, e.g. 
    from Scan or Contours.
*/

class MnPlot {

public:
  
  MnPlot() : thePageWidth(80), thePageLength(30) {}

  MnPlot(unsigned int width, unsigned int length) : thePageWidth(width), thePageLength(length) {
    if(thePageWidth > 120) thePageWidth = 120;
    if(thePageLength > 56) thePageLength = 56;
  }

  ~MnPlot() {}

  void operator()(const std::vector<std::pair<double,double> >&) const;
  void operator()(double, double, const std::vector<std::pair<double,double> >&) const;

  unsigned int width() const {return thePageWidth;}
  unsigned int length() const {return thePageLength;}

private:

  unsigned int thePageWidth;
  unsigned int thePageLength;
};

#endif //MN_MnPlot_H_
