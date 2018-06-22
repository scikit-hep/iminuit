"""Minuit C++ class to IMinuit Python struct mappings.
"""

cdef cfmin2struct(FunctionMinimum*cfmin):
    cfmin_struct = Struct(
        fval=cfmin.Fval(),
        edm=cfmin.Edm(),
        nfcn=cfmin.NFcn(),
        up=cfmin.Up(),
        is_valid=cfmin.IsValid(),
        has_valid_parameters=cfmin.HasValidParameters(),
        has_accurate_covar=cfmin.HasAccurateCovar(),
        has_posdef_covar=cfmin.HasPosDefCovar(),
        #forced to be posdef
        has_made_posdef_covar=cfmin.HasMadePosDefCovar(),
        hesse_failed=cfmin.HesseFailed(),
        has_covariance=cfmin.HasCovariance(),
        is_above_max_edm=cfmin.IsAboveMaxEdm(),
        has_reached_call_limit=cfmin.HasReachedCallLimit()
    )
    return cfmin_struct


cdef minuitparam2struct(MinuitParameter mp):
    ret = Struct(
        number=mp.Number(),
        name=mp.Name(),
        value=mp.Value(),
        error=mp.Error(),
        is_const=mp.IsConst(),
        is_fixed=mp.IsFixed(),
        has_limits=mp.HasLimits(),
        has_lower_limit=mp.HasLowerLimit(),
        has_upper_limit=mp.HasUpperLimit(),
        lower_limit=mp.LowerLimit() if mp.HasLowerLimit() else None,
        upper_limit=mp.UpperLimit() if mp.HasUpperLimit() else None,
    )
    return ret


cdef minoserror2struct(MinosError m):
    ret = Struct(
        lower=m.Lower(),
        upper=m.Upper(),
        is_valid=m.IsValid(),
        lower_valid=m.LowerValid(),
        upper_valid=m.UpperValid(),
        at_lower_limit=m.AtLowerLimit(),
        at_upper_limit=m.AtUpperLimit(),
        at_lower_max_fcn=m.AtLowerMaxFcn(),
        at_upper_max_fcn=m.AtUpperMaxFcn(),
        lower_new_min=m.LowerNewMin(),
        upper_new_min=m.UpperNewMin(),
        nfcn=m.NFcn(),
        min=m.Min()
    )
    return ret
