#include "Minuit/MnUserParameters.h"

MnUserParameters::MnUserParameters(const std::vector<double>& par, const std::vector<double>& err) : theTransformation(par, err) {}

const std::vector<MinuitParameter>& MnUserParameters::parameters() const {
  return theTransformation.parameters();
}

std::vector<double> MnUserParameters::params() const {
  return theTransformation.params();
}

std::vector<double> MnUserParameters::errors() const {
  return theTransformation.errors();
}

const MinuitParameter& MnUserParameters::parameter(unsigned int n) const {
  return theTransformation.parameter(n);
}

bool MnUserParameters::add(const char* name, double val, double err) {
  return theTransformation.add(name, val, err);
}

bool  MnUserParameters::add(const char* name, double val, double err, double low, double up) {
  return theTransformation.add(name, val, err, low, up);
}

bool  MnUserParameters::add(const char* name, double val) {
  return theTransformation.add(name, val);
}

void MnUserParameters::fix(unsigned int n) {
  theTransformation.fix(n);
}

void MnUserParameters::release(unsigned int n) {
  theTransformation.release(n);
}

void MnUserParameters::setValue(unsigned int n, double val) {
  theTransformation.setValue(n, val);
}

void MnUserParameters::setError(unsigned int n, double err) {
  theTransformation.setError(n, err);
}

void MnUserParameters::setLimits(unsigned int n, double low, double up) {
  theTransformation.setLimits(n, low, up);
}

void MnUserParameters::setUpperLimit(unsigned int n, double up) {
  theTransformation.setUpperLimit(n, up);
}

void MnUserParameters::setLowerLimit(unsigned int n, double low) {
  theTransformation.setLowerLimit(n, low);
}

void MnUserParameters::removeLimits(unsigned int n) {
  theTransformation.removeLimits(n);
}

double MnUserParameters::value(unsigned int n) const {
  return theTransformation.value(n);
}

double MnUserParameters::error(unsigned int n) const {
  return theTransformation.error(n);
}

void MnUserParameters::fix(const char* name) {
  fix(index(name));
}

void MnUserParameters::release(const char* name) {
  release(index(name));
}

void MnUserParameters::setValue(const char* name, double val) {
  setValue(index(name), val);
}

void MnUserParameters::setError(const char* name, double err) {
  setError(index(name), err);
}

void MnUserParameters::setLimits(const char* name, double low, double up) {
  setLimits(index(name), low, up);
}

void MnUserParameters::setUpperLimit(const char* name, double up) {
  theTransformation.setUpperLimit(index(name), up);
}

void MnUserParameters::setLowerLimit(const char* name, double low) {
  theTransformation.setLowerLimit(index(name), low);
}

void MnUserParameters::removeLimits(const char* name) {
  removeLimits(index(name));
}

double MnUserParameters::value(const char* name) const {
  return value(index(name));
}

double MnUserParameters::error(const char* name) const {
  return error(index(name));
}
  
unsigned int MnUserParameters::index(const char* name) const {
  return theTransformation.index(name);
}

const char* MnUserParameters::name(unsigned int n) const {
  return theTransformation.name(n);
}

const MnMachinePrecision& MnUserParameters::precision() const {
  return theTransformation.precision();
}
