"""Minuit C++ class to IMinuit Python struct mappings.
"""
from iminuit.util import FMin, Param, MError

cdef cfmin2struct(FunctionMinimum* cfmin, tolerance, ncalls):
    return FMin(cfmin.Fval(), cfmin.Edm(), tolerance, cfmin.NFcn(), ncalls,
         cfmin.Up(), cfmin.IsValid(), cfmin.HasValidParameters(),
         cfmin.HasAccurateCovar(), cfmin.HasPosDefCovar(), cfmin.HasMadePosDefCovar(),
         cfmin.HesseFailed(), cfmin.HasCovariance(), cfmin.IsAboveMaxEdm(),
         cfmin.HasReachedCallLimit())


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
