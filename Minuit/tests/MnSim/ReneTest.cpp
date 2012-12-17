// $Id: ReneTest.cpp,v 1.12 2005/01/20 11:19:57 moneta Exp $
#ifdef _WIN32 
  #define _USE_MATH_DEFINES
#endif
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnMinos.h"
#include "Minuit/MnUserParameters.h"
#include "Minuit/MnPrint.h"
#include "Minuit/FCNBase.h"
#include "Minuit/MnScan.h"
#include "Minuit/MnPlot.h"

class ReneFcn : public FCNBase {

public:

  ReneFcn(const std::vector<double>& meas) : theMeasurements(meas) {}

  virtual ~ReneFcn() {}

  virtual double operator()(const std::vector<double>& par) const {
    double a = par[2];
    double b = par[1];
    double c = par[0];
    double p0 = par[3];
    double p1 = par[4];
    double p2 = par[5];
    double fval = 0.;
//     double nbin = 3./double(theMeasurements.size());
    for(unsigned int i = 0; i < theMeasurements.size(); i++) {
      double ni = theMeasurements[i];
      if(ni < 1.e-10) continue;
//       double xi = (i + 0.5)*nbin; //xi=0-3
//       double xi = (i+0.5)/20.; //xi=0-3
//       double xi = (i+0.5)/40.; //xi=0-3
      double xi = (i+1.)/40. - 1./80.; //xi=0-3
      double ei = ni;
//       double nexp1 = a*xi*xi + b*xi + c;
//       double nexp2 = 0.5*p0*p1/M_PI;
//       double nexp3 = std::max(1.e-10, (xi-p2)*(xi-p2) + 0.25*p1*p1); 
//       double nexp = nexp1 + nexp2/nexp3;
      double nexp = a*xi*xi + b*xi + c + (0.5*p0*p1/M_PI)/std::max(1.e-10, (xi-p2)*(xi-p2) + 0.25*p1*p1);
      fval += (ni-nexp)*(ni-nexp)/ei;
    }
    return fval;
  }

  virtual double up() const {return 1.;}

private:
  std::vector<double> theMeasurements;
};

 
/*
extern "C" void fcnr_(int&, double[], double&, double[], int&);
extern "C" void stand_() {}

class ReneFcn : public FCNBase {

public:

  ReneFcn(const std::vector<double>& meas) : theMeasurements(meas) {}

  virtual ~ReneFcn() {}

  virtual double operator()(const std::vector<double>& par) const {
    double mypar[6];
    for(std::vector<double>::const_iterator ipar = par.begin();
	ipar != par.end(); ipar++)
      mypar[ipar-par.begin()] = par[ipar-par.begin()];
    double fval = 0.;
    int iflag = 4;
    int npar = par.size();
    fcnr_(npar, 0,  fval, mypar, iflag);
    
    return fval;
  }

  virtual double up() const {return 1.;}

private:
  std::vector<double> theMeasurements;
};

*/


int main() {
  /*
  double tmp[60] = {6., 1.,10.,12., 6.,13.,23.,22.,15.,21.,
		    23.,26.,36.,25.,27.,35.,40.,44.,66.,81.,
		    75.,57.,48.,45.,46.,41.,35.,36.,53.,32.,
		    40.,37.,38.,31.,36.,44.,42.,37.,32.,32.,
		    43.,44.,35.,33.,33.,39.,29.,41.,32.,44.,
		    26.,39.,29.,35.,32.,21.,21.,15.,25.,15.};
  std::vector<double> measurements(tmp, tmp+60);
  */
  /*
  double tmp[120] = {2.,1.,1.,0.,1.,1.,0.,1.,3.,0.,0.,1.,0.,
		     1.,1.,0.,0.,1.,0.,0.,0.,0.,2.,1.,1.,2.,
		     2.,0.,2.,4.,2.,6.,2.,1.,4.,0.,3.,6.,16.,
		     30.,34.,18.,8.,2.,3.,4.,4.,5.,6.,3.,5.,
		     0.,1.,1.,7.,3.,2.,5.,1.,3.,5.,3.,2.,3.,
		     2.,2.,1.,1.,5.,2.,3.,7.,2.,7.,6.,5.,1.,
		     4.,5.,0.,6.,3.,4.,3.,3.,6.,8.,8.,3.,4.,
		     4.,8.,9.,7.,3.,4.,6.,2.,5.,10.,7.,6.,4.,
		     4.,7.,7.,5.,4.,12.,4.,6.,3.,7.,4.,3.,4.,
		     3,10.,8.,7.};  
  */
  double tmp[120] = {38.,36.,46.,52.,54.,52.,61.,52.,64.,77.,
		     60.,56.,78.,71.,81.,83.,89.,96.,118.,96.,
		     109.,111.,107.,107.,135.,156.,196.,137.,
		     160.,153.,185.,222.,251.,270.,329.,422.,
		     543.,832.,1390.,2835.,3462.,2030.,1130.,
		     657.,469.,411.,375.,295.,281.,281.,289.,
		     273.,297.,256.,274.,287.,280.,274.,286.,
		     279.,293.,314.,285.,322.,307.,313.,324.,
		     351.,314.,314.,301.,361.,332.,342.,338.,
		     396.,356.,344.,395.,416.,406.,411.,422.,
		     393.,393.,409.,455.,427.,448.,459.,403.,
		     441.,510.,501.,502.,482.,487.,506.,506.,
		     526.,517.,534.,509.,482.,591.,569.,518.,
		     609.,569.,598.,627.,617.,610.,662.,666.,
		     652.,671.,647.,650.,701.};

  std::vector<double> measurements(tmp, tmp+120);
					       		     
  ReneFcn theFCN(measurements);

  MnUserParameters upar;
  upar.add("p0", 100., 10.);
  upar.add("p1", 100., 10.);
  upar.add("p2", 100., 10.);
  upar.add("p3", 100., 10.);
  upar.add("p4", 1., 0.3);
  upar.add("p5", 1., 0.3);
  /*
# ext. ||   name    ||   type  ||   value   ||  error +/- 

   0   ||        p0 ||  free   ||     32.04 ||   9.611
   1   ||        p1 ||  free   ||     98.11 ||   29.43
   2   ||        p2 ||  free   ||     39.15 ||   11.75
   3   ||        p3 ||  free   ||     362.4 ||   108.7
   4   ||        p4 ||  free   ||   0.06707 || 0.02012
   5   ||        p5 ||  free   ||     1.006 ||  0.3019

  upar.add(0, "p0", 32.04, 9.611);
  upar.add(1, "p1", 98.11, 29.43);
  upar.add(2, "p2", 39.15, 11.75);
  upar.add(3, "p3", 362.4, 108.7);
  upar.add(4, "p4", 0.06707, 0.02012);
  upar.add(5, "p5", 1.006, 0.3019);
  */

  std::cout<<"initial parameters: "<<upar<<std::endl;

  std::cout<<"start migrad "<<std::endl;
  MnMigrad migrad(theFCN, upar);
  FunctionMinimum min = migrad();
  if(!min.isValid()) {
    //try with higher strategy
    std::cout<<"FM is invalid, try with strategy = 2."<<std::endl;
    MnMigrad migrad2(theFCN, min.userState(), MnStrategy(2));
    min = migrad2();
  } 
  std::cout<<"minimum: "<<min<<std::endl;
  /*
  std::cout<<"start minos"<<std::endl;
  MnMinos minos(migrad, min);
  AsymmetricError e0 = minos(0);
  AsymmetricError e1 = minos(1);
  AsymmetricError e2 = minos(2);
  
  std::cout<<"par0: "<<e0.value()<<" + "<<e0.upper()<<e0.lower()<<std::endl;
  std::cout<<"par1: "<<e1.value()<<" + "<<e1.upper()<<e1.lower()<<std::endl;
  std::cout<<"par2: "<<e2.value()<<" + "<<e2.upper()<<e2.lower()<<std::endl;
  */

  {
    std::vector<double> params(6, 1.);
    std::vector<double> error(6, 1.);
    MnScan scan(theFCN, params, error);
    std::cout<<"scan parameters: "<<scan.parameters()<<std::endl;
    MnPlot plot;
    for(unsigned int i = 0; i < upar.variableParameters(); i++) {
      std::vector<std::pair<double, double> > xy = scan.scan(i);
//       std::vector<std::pair<double, double> > xy = scan.scan(0);
      plot(xy);
    }
    std::cout<<scan.parameters()<<std::endl;
  }

  {
    std::vector<double> params(6, 1.);
    std::vector<double> error(6, 1.);
    MnScan scan(theFCN, params, error);
    std::cout<<"scan parameters: "<<scan.parameters()<<std::endl;
    FunctionMinimum min = scan();
//     std::cout<<min<<std::endl;
    std::cout<<scan.parameters()<<std::endl;
  }

  return 0;
}

