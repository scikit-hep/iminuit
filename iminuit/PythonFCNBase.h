#ifndef PYTHONFCNBASE
#define PYTHONFCNBASE

class PythonFCNBase {
    public:virtual ~PythonFCNBase() {}
    virtual double operator()(const std::vector<double>& x) const = 0;
    virtual double ErrorDef() const {return Up();}
    virtual double Up() const = 0;
    virtual void SetErrorDef(double ) {};

    int getNumCall() const {return 0;};
    void set_up(double up) {};
    void resetNumCall() {};
};

#endif  // PYTHONFCNBASE
