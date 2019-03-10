"""Minuit C++ class to IMinuit Python struct mappings.
"""
from iminuit.util import Struct, FMin, Param

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


cdef minoserror2struct(MinosError m):
    return Struct(
        [("is_valid", m.IsValid()),
         ("lower", m.Lower()),
         ("upper", m.Upper()),
         ("lower_valid", m.LowerValid()),
         ("upper_valid", m.UpperValid()),
         ("at_lower_limit", m.AtLowerLimit()),
         ("at_upper_limit", m.AtUpperLimit()),
         ("at_lower_max_fcn", m.AtLowerMaxFcn()),
         ("at_upper_max_fcn", m.AtUpperMaxFcn()),
         ("lower_new_min", m.LowerNewMin()),
         ("upper_new_min", m.UpperNewMin()),
         ("nfcn", m.NFcn()),
         ("min", m.Min())],
    )
