#ifndef MN_ContoursError_H_
#define MN_ContoursError_H_

#include "Minuit/MnConfig.h"
#include "Minuit/MinosError.h"

#include <vector>
#include <utility>

class ContoursError {

public:

    ContoursError():
        theParX(), theParY(),
        thePoints(), theXMinos(), theYMinos(),
        theNFcn(){}

    ContoursError(unsigned int parx, unsigned int pary,
                const std::vector<std::pair<double,double> >& points,
                const MinosError& xmnos, const MinosError& ymnos,
                unsigned int nfcn) :
                theParX(parx), theParY(pary),
                thePoints(points), theXMinos(xmnos), theYMinos(ymnos),
                theNFcn(nfcn) {}

    ~ContoursError() {}

    ContoursError(const ContoursError& cont) :
        theParX(cont.theParX),
        theParY(cont.theParY),
        thePoints(cont.thePoints),
        theXMinos(cont.theXMinos),
        theYMinos(cont.theYMinos),
        theNFcn(cont.theNFcn) {}

    ContoursError& operator()(const ContoursError& cont) {
        theParX = cont.theParX;
        theParY = cont.theParY;
        thePoints = cont.thePoints;
        theXMinos = cont.theXMinos;
        theYMinos = cont.theYMinos;
        theNFcn = cont.theNFcn;
        return *this;
    }

    const std::vector<std::pair<double,double> >& operator()() const {
        return thePoints;
    }

    std::vector<std::pair<double,double> > points() const {//return a copy
        return thePoints;
    }

    std::pair<double,double> xMinos() const {
        return theXMinos();
    }

    std::pair<double,double> yMinos() const {
        return theYMinos();
    }

    unsigned int xpar() const {return theParX;}
    unsigned int ypar() const {return theParY;}

    const MinosError& xMinosError() const {
        return theXMinos;
    }

    const MinosError& yMinosError() const {
        return theYMinos;
    }

    unsigned int nfcn() const {return theNFcn;}
    double xmin() const {return theXMinos.min();}
    double ymin() const {return theYMinos.min();}

private:

    unsigned int theParX;
    unsigned int theParY;
    std::vector<std::pair<double,double> > thePoints;
    MinosError theXMinos;
    MinosError theYMinos;
    unsigned int theNFcn;
};

#endif //MN_ContoursError_H_
