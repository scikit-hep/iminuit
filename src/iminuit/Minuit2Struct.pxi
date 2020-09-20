"""Minuit C++ class to IMinuit Python struct mappings.
"""
from iminuit.util import FMin, Param, MError

cdef cfmin2struct(FunctionMinimum* cfmin, tolerance, ncalls, ncalls_total, ngrads, ngrads_total):
    has_parameters_at_limit = False
    cdef double v, e, l, u
    cdef int i
    cdef const MinuitParameter* mp
    for i in range(cfmin.UserState().MinuitParameters().size()):
        mp = &cfmin.UserState().MinuitParameters()[i]
        if not mp.HasLimits():
            continue
        v = mp.Value()
        e = mp.Error()
        l = mp.LowerLimit()
        u = mp.UpperLimit()
        # the 0.5 error threshold is somewhat arbitrary
        has_parameters_at_limit |= min(v - l, u - v) < 0.5 * e

    return FMin(cfmin.Fval(), cfmin.Edm(), tolerance, ncalls, ncalls_total,
         cfmin.Up(), cfmin.IsValid(), cfmin.HasValidParameters(),
         cfmin.HasAccurateCovar(), cfmin.HasPosDefCovar(), cfmin.HasMadePosDefCovar(),
         cfmin.HesseFailed(), cfmin.HasCovariance(), cfmin.IsAboveMaxEdm(),
         cfmin.HasReachedCallLimit(), has_parameters_at_limit, ngrads, ngrads_total)


cdef minuitparam2struct(MinuitParameter mp):
    return Param(mp.Number(), mp.Name(), mp.Value(), mp.Error(), mp.IsConst(),
                 mp.IsFixed(), mp.HasLimits(), mp.HasLowerLimit(), mp.HasUpperLimit(),
                 mp.LowerLimit() if mp.HasLowerLimit() else None,
                 mp.UpperLimit() if mp.HasUpperLimit() else None)


cdef minoserror2struct(name, MinosError m):
    return MError(name, m.IsValid(), m.Lower(), m.Upper(),
      m.LowerValid(), m.UpperValid(), m.AtLowerLimit(), m.AtUpperLimit(),
      m.AtLowerMaxFcn(), m.AtUpperMaxFcn(), m.LowerNewMin(), m.UpperNewMin(),
      m.NFcn(), m.Min())
