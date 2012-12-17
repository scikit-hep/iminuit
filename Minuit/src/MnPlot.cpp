#include "Minuit/MnPlot.h"

void mnplot(double* xpt, double* ypt, char* chpt, int nxypt, int npagwd, int npagln);

void MnPlot::operator()(const std::vector<std::pair<double,double> >& points) const {
  std::vector<double> x; x.reserve(points.size());
  std::vector<double> y; y.reserve(points.size());
  std::vector<char> chpt; chpt.reserve(points.size());
  
  for(std::vector<std::pair<double,double> >::const_iterator ipoint = points.begin(); ipoint != points.end(); ipoint++) {
    x.push_back((*ipoint).first);
    y.push_back((*ipoint).second);
    chpt.push_back('*');
  }

  mnplot(&(x.front()), &(y.front()), &(chpt.front()), points.size(), width(), length());

}

void MnPlot::operator()(double xmin, double ymin, const std::vector<std::pair<double,double> >& points) const {
  std::vector<double> x; x.reserve(points.size()+2);
  x.push_back(xmin);
  x.push_back(xmin);
  std::vector<double> y; y.reserve(points.size()+2);
  y.push_back(ymin);
  y.push_back(ymin);
  std::vector<char> chpt; chpt.reserve(points.size()+2);
  chpt.push_back(' ');
  chpt.push_back('X');
  
  for(std::vector<std::pair<double,double> >::const_iterator ipoint = points.begin(); ipoint != points.end(); ipoint++) {
    x.push_back((*ipoint).first);
    y.push_back((*ipoint).second);
    chpt.push_back('*');
  }

  mnplot(&(x.front()), &(y.front()), &(chpt.front()), points.size()+2, width(), length());
}
