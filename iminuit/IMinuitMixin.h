#ifndef IMINUIT_IMINUIT_MIXIN_H
#define IMINUIT_IMINUIT_MIXIN_H

class IMinuitMixin {
public:
    IMinuitMixin() {}

    IMinuitMixin(const IMinuitMixin& x) :
        up(x.up), names(x.names), throw_nan(x.throw_nan)
    {}

    IMinuitMixin(double u,
                 const std::vector<std::string>& pn,
                 bool t) :
        up(u), names(pn), throw_nan(t)
    {}

    virtual ~IMinuitMixin() {}

    virtual int getNumCall() const = 0;
    virtual void resetNumCall() = 0;

protected:
    double up;
    std::vector<std::string> names;
    bool throw_nan;
};

#endif
