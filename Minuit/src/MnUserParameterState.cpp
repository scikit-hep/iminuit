#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnCovarianceSqueeze.h"
#include "Minuit/MinimumState.h"

//
// construct from user parameters (befor minimization)
//
MnUserParameterState::MnUserParameterState(const std::vector<double>& par, const std::vector<double>& err) : theValid(true), theCovarianceValid(false), theGCCValid(false), theFVal(0.), theEDM(0.), theNFcn(0), theParameters(MnUserParameters(par, err)), theCovariance(MnUserCovariance()), theGlobalCC(MnGlobalCorrelationCoeff()), theIntParameters(par), theIntCovariance(MnUserCovariance()) {}

MnUserParameterState::MnUserParameterState(const MnUserParameters& par) : theValid(true), theCovarianceValid(false), theGCCValid(false), theFVal(0.), theEDM(0.), theNFcn(0), theParameters(par), theCovariance(MnUserCovariance()), theGlobalCC(MnGlobalCorrelationCoeff()), theIntParameters(std::vector<double>()), theIntCovariance(MnUserCovariance()) {

    for(std::vector<MinuitParameter>::const_iterator ipar = minuitParameters().begin(); ipar != minuitParameters().end(); ipar++) {
        if((*ipar).isConst() || (*ipar).isFixed()) continue;
        if((*ipar).hasLimits())
            theIntParameters.push_back(ext2int((*ipar).number(), (*ipar).value()));
        else
            theIntParameters.push_back((*ipar).value());
    }
}

//
// construct from user parameters + errors (befor minimization)
//
MnUserParameterState::MnUserParameterState(const std::vector<double>& par, const std::vector<double>& cov, unsigned int nrow) : theValid(true), theCovarianceValid(true), theGCCValid(false), theFVal(0.), theEDM(0.), theNFcn(0), theParameters(MnUserParameters()), theCovariance(MnUserCovariance(cov, nrow)), theGlobalCC(MnGlobalCorrelationCoeff()), theIntParameters(par), theIntCovariance(MnUserCovariance(cov, nrow)) {
    std::vector<double> err; err.reserve(par.size());
    for(unsigned int i = 0; i < par.size(); i++) {
        assert(theCovariance(i,i) > 0.);
        err.push_back(sqrt(theCovariance(i,i)));
    }
    theParameters = MnUserParameters(par, err);
    assert(theCovariance.nrow() == variableParameters());
}

MnUserParameterState::MnUserParameterState(const std::vector<double>& par, const MnUserCovariance& cov) : theValid(true), theCovarianceValid(true), theGCCValid(false), theFVal(0.), theEDM(0.), theNFcn(0), theParameters(MnUserParameters()), theCovariance(cov), theGlobalCC(MnGlobalCorrelationCoeff()), theIntParameters(par), theIntCovariance(cov) {
    std::vector<double> err; err.reserve(par.size());
    for(unsigned int i = 0; i < par.size(); i++) {
        assert(theCovariance(i,i) > 0.);
        err.push_back(sqrt(theCovariance(i,i)));
    }
    theParameters = MnUserParameters(par, err);
    assert(theCovariance.nrow() == variableParameters());
}


MnUserParameterState::MnUserParameterState(const MnUserParameters& par, const MnUserCovariance& cov) : theValid(true), theCovarianceValid(true), theGCCValid(false), theFVal(0.), theEDM(0.), theNFcn(0), theParameters(par), theCovariance(cov), theGlobalCC(MnGlobalCorrelationCoeff()), theIntParameters(std::vector<double>()), theIntCovariance(cov) {
    theIntCovariance.scale(0.5);
    for(std::vector<MinuitParameter>::const_iterator ipar = minuitParameters().begin(); ipar != minuitParameters().end(); ipar++) {
        if((*ipar).isConst() || (*ipar).isFixed()) continue;
        if((*ipar).hasLimits()) 
            theIntParameters.push_back(ext2int((*ipar).number(), (*ipar).value()));
        else 
            theIntParameters.push_back((*ipar).value());
    }
    assert(theCovariance.nrow() == variableParameters());
//
// need to fix that in case of limited parameters
//   theIntCovariance = MnUserCovariance();
//
}

//
// construct from internal parameters (after minimization)
//
MnUserParameterState::MnUserParameterState(const MinimumState& st, double up, const MnUserTransformation& trafo) : theValid(st.isValid()), theCovarianceValid(false), theGCCValid(false), theFVal(st.fval()), theEDM(st.edm()), theNFcn(st.nfcn()), theParameters(MnUserParameters()), theCovariance(MnUserCovariance()), theGlobalCC(MnGlobalCorrelationCoeff()), theIntParameters(std::vector<double>()), theIntCovariance(MnUserCovariance()) {

    for(std::vector<MinuitParameter>::const_iterator ipar = trafo.parameters().begin(); ipar != trafo.parameters().end(); ipar++) {
        if((*ipar).isConst()) {
            add((*ipar).name(), (*ipar).value());
        } else if((*ipar).isFixed()) {
            add((*ipar).name(), (*ipar).value(), (*ipar).error());
            if((*ipar).hasLimits()) {
                if((*ipar).hasLowerLimit() && (*ipar).hasUpperLimit())
                    setLimits((*ipar).name(), (*ipar).lowerLimit(),(*ipar).upperLimit());
                else if((*ipar).hasLowerLimit() && !(*ipar).hasUpperLimit())
                    setLowerLimit((*ipar).name(), (*ipar).lowerLimit());
                else
                    setUpperLimit((*ipar).name(), (*ipar).upperLimit());
            }
            fix((*ipar).name());
        } else if((*ipar).hasLimits()) {
            unsigned int i = trafo.intOfExt((*ipar).number());
            double err = st.hasCovariance() ? sqrt(2.*up*st.error().invHessian()(i,i)) : st.parameters().dirin()(i);
            add((*ipar).name(), trafo.int2ext(i, st.vec()(i)), trafo.int2extError(i, st.vec()(i), err));
            if((*ipar).hasLowerLimit() && (*ipar).hasUpperLimit())
                setLimits((*ipar).name(), (*ipar).lowerLimit(), (*ipar).upperLimit());
            else if((*ipar).hasLowerLimit() && !(*ipar).hasUpperLimit())
                setLowerLimit((*ipar).name(), (*ipar).lowerLimit());
            else
                setUpperLimit((*ipar).name(), (*ipar).upperLimit());
        } else {
            unsigned int i = trafo.intOfExt((*ipar).number());
            double err = st.hasCovariance() ? sqrt(2.*up*st.error().invHessian()(i,i)) : st.parameters().dirin()(i);
            add((*ipar).name(), st.vec()(i), err);
        }
    }

    theCovarianceValid = st.error().isValid();

    if(theCovarianceValid) {
        theCovariance = trafo.int2extCovariance(st.vec(), st.error().invHessian());
        theIntCovariance = MnUserCovariance(std::vector<double>(st.error().invHessian().data(), st.error().invHessian().data()+st.error().invHessian().size()), st.error().invHessian().nrow());
        theCovariance.scale(2.*up);
        theGlobalCC = MnGlobalCorrelationCoeff(st.error().invHessian());
        theGCCValid = true;

        assert(theCovariance.nrow() == variableParameters());
    }
}

// facade: forward interface of MnUserParameters and MnUserTransformation
// via MnUserParameterState

//access to parameters (row-wise)
const std::vector<MinuitParameter>& MnUserParameterState::minuitParameters() const {
    return theParameters.parameters();
}
//access to parameters and errors in column-wise representation 
std::vector<double> MnUserParameterState::params() const {
    return theParameters.params();
}
std::vector<double> MnUserParameterState::errors() const {
    return theParameters.errors();
}

//access to single parameter
const MinuitParameter& MnUserParameterState::parameter(unsigned int i) const {
    return theParameters.parameter(i);
}

//add free parameter
void MnUserParameterState::add(const char* name, double val, double err) {
    if ( theParameters.add(name, val, err) ) { 
        theIntParameters.push_back(val);
        theCovarianceValid = false;
        theGCCValid = false;
        theValid = true;
    }
    else { 
        int i = index(name);
        setValue(i,val);
        setError(i,err);
    }

}

//add limited parameter
void MnUserParameterState::add(const char* name, double val, double err, double low, double up) {
    if ( theParameters.add(name, val, err, low, up) ) {  
        theCovarianceValid = false;
        theIntParameters.push_back(ext2int(index(name), val));
        theGCCValid = false;
        theValid = true;
    }
else { // parameter already exist - just set values
    int i = index(name);
    setValue(i,val);
    setError(i,err);
    setLimits(i,low,up);
}
}

//add const parameter
void MnUserParameterState::add(const char* name, double val) {
    if ( theParameters.add(name, val) )
        theValid = true;
    else
        setValue(name,val);
}

//interaction via external number of parameter
void MnUserParameterState::fix(unsigned int e) {
    unsigned int i = intOfExt(e);
    if(theCovarianceValid) {
        theCovariance = MnCovarianceSqueeze()(theCovariance, i);
        theIntCovariance = MnCovarianceSqueeze()(theIntCovariance, i);
    }
    theIntParameters.erase(theIntParameters.begin()+i, theIntParameters.begin()+i+1);
    theParameters.fix(e);
    theGCCValid = false;
}

void MnUserParameterState::release(unsigned int e) {
    theParameters.release(e);
    theCovarianceValid = false;
    theGCCValid = false;
    unsigned int i = intOfExt(e);
    if(parameter(e).hasLimits())
        theIntParameters.insert(theIntParameters.begin()+i, ext2int(e, parameter(e).value()));
    else
        theIntParameters.insert(theIntParameters.begin()+i, parameter(e).value());
}

void MnUserParameterState::setValue(unsigned int e, double val) {
    theParameters.setValue(e, val);
    if(!parameter(e).isFixed() && !parameter(e).isConst()) {
        unsigned int i = intOfExt(e);
        if(parameter(e).hasLimits())
            theIntParameters[i] = ext2int(e, val);
        else
            theIntParameters[i] = val;
    }
}

void MnUserParameterState::setError(unsigned int e, double val) {
    theParameters.setError(e, val);
}

void MnUserParameterState::setLimits(unsigned int e, double low, double up) {
    theParameters.setLimits(e, low, up);
    theCovarianceValid = false;
    theGCCValid = false;
    if(!parameter(e).isFixed() && !parameter(e).isConst()) {
        unsigned int i = intOfExt(e);
        if(low < theIntParameters[i] && theIntParameters[i] < up)
            theIntParameters[i] = ext2int(e, theIntParameters[i]);
        else
            theIntParameters[i] = ext2int(e, 0.5*(low+up));
    }
}

void MnUserParameterState::setUpperLimit(unsigned int e, double up) {
    theParameters.setUpperLimit(e, up);
    theCovarianceValid = false;
    theGCCValid = false;
    if(!parameter(e).isFixed() && !parameter(e).isConst()) {
        unsigned int i = intOfExt(e);
        if(theIntParameters[i] < up)
            theIntParameters[i] = ext2int(e, theIntParameters[i]);
        else
            theIntParameters[i] = ext2int(e, up - 0.5*fabs(up + 1.));
    }
}

void MnUserParameterState::setLowerLimit(unsigned int e, double low) {
    theParameters.setLowerLimit(e, low);
    theCovarianceValid = false;
    theGCCValid = false;
    if(!parameter(e).isFixed() && !parameter(e).isConst()) {
        unsigned int i = intOfExt(e);
        if(low < theIntParameters[i])
            theIntParameters[i] = ext2int(e, theIntParameters[i]);
        else
            theIntParameters[i] = ext2int(e, low + 0.5*fabs(low + 1.));
    }
}

void MnUserParameterState::removeLimits(unsigned int e) {
    theParameters.removeLimits(e);
    theCovarianceValid = false;
    theGCCValid = false;
    if(!parameter(e).isFixed() && !parameter(e).isConst())
        theIntParameters[intOfExt(e)] = value(e);  
}

double MnUserParameterState::value(unsigned int i) const {
    return theParameters.value(i);
}
double MnUserParameterState::error(unsigned int i) const {
    return theParameters.error(i);
}

//interaction via name of parameter
void MnUserParameterState::fix(const char* name) {
    fix(index(name));
}

void MnUserParameterState::release(const char* name) {
    release(index(name));
}

void MnUserParameterState::setValue(const char* name, double val) {
    setValue(index(name), val);
}

void MnUserParameterState::setError(const char* name, double val) {
    setError(index(name), val);
}

void MnUserParameterState::setLimits(const char* name, double low, double up) {
    setLimits(index(name), low, up);
}

void MnUserParameterState::setUpperLimit(const char* name, double up) {
    setUpperLimit(index(name), up);
}

void MnUserParameterState::setLowerLimit(const char* name, double low) {
    setLowerLimit(index(name), low);
}

void MnUserParameterState::removeLimits(const char* name) {
    removeLimits(index(name));
}

double MnUserParameterState::value(const char* name) const {
    return value(index(name));
}
double MnUserParameterState::error(const char* name) const {
    return error(index(name));
}

//convert name into external number of parameter
unsigned int MnUserParameterState::index(const char* name) const {
    return theParameters.index(name);
}
//convert external number into name of parameter
const char* MnUserParameterState::name(unsigned int i) const {
    return theParameters.name(i);
}

// transformation internal <-> external
double MnUserParameterState::int2ext(unsigned int i, double val) const {
    return theParameters.trafo().int2ext(i, val);
}
double MnUserParameterState::ext2int(unsigned int e, double val) const {
    return theParameters.trafo().ext2int(e, val);
}
unsigned int MnUserParameterState::intOfExt(unsigned int ext) const {
    return theParameters.trafo().intOfExt(ext);
}
unsigned int MnUserParameterState::extOfInt(unsigned int internal) const { 
    return theParameters.trafo().extOfInt(internal);
}
unsigned int MnUserParameterState::variableParameters() const {
    return theParameters.trafo().variableParameters();
}
const MnMachinePrecision& MnUserParameterState::precision() const {
    return theParameters.precision();
}

void MnUserParameterState::setPrecision(double eps) {
    theParameters.setPrecision(eps);
}
