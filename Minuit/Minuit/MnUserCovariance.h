#ifndef MN_MnUserCovariance_H_
#define MN_MnUserCovariance_H_

#include "Minuit/MnConfig.h"
#include <vector>
#include <cassert>

class MnUserCovariance {

public:

    MnUserCovariance() : theData(std::vector<double>()), theNRow(0) {}

    MnUserCovariance(const std::vector<double>& data, unsigned int nrow) :
    theData(data), theNRow(nrow) {
        assert(data.size() == nrow*(nrow+1)/2);
    }

    MnUserCovariance(unsigned int n) :
    theData(std::vector<double>(n*(n+1)/2, 0.)), theNRow(n) {}

    ~MnUserCovariance() {}

    MnUserCovariance(const MnUserCovariance& cov) : theData(cov.theData), theNRow(cov.theNRow) {}

    MnUserCovariance& operator=(const MnUserCovariance& cov) {
        theData = cov.theData;
        theNRow = cov.theNRow;
        return *this;
    }

    double operator()(unsigned int row, unsigned int col) const {
        assert(row < theNRow && col < theNRow);
        if(row > col)
            return theData[col+row*(row+1)/2];
        else
            return theData[row+col*(col+1)/2];
    }

    double& operator()(unsigned int row, unsigned int col) {
        assert(row < theNRow && col < theNRow);
        if(row > col)
            return theData[col+row*(row+1)/2];
        else
            return theData[row+col*(col+1)/2];
    }

    void scale(double f) {
        for(unsigned int i = 0; i < theData.size(); i++) theData[i] *= f;
    }

const std::vector<double>& data() const {return theData;}

unsigned int nrow() const {return theNRow;}

// VC 7.1 warning: conversion from size_t to unsigned int
unsigned int size() const 
{ return static_cast < unsigned int > ( theData.size() );
}

private:

    std::vector<double> theData;
    unsigned int theNRow;
};

#endif //MN_MnUserCovariance_H_
