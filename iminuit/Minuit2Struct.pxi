cdef cfmin2struct(FunctionMinimum* cfmin):
    cfmin_struct = Struct(
            fval = cfmin.fval(),
            edm = cfmin.edm(),
            nfcn = cfmin.nfcn(),
            up = cfmin.up(),
            is_valid = cfmin.isValid(),
            has_valid_parameters = cfmin.hasValidParameters(),
            has_accurate_covar = cfmin.hasAccurateCovar(),
            has_posdef_covar = cfmin.hasPosDefCovar(),
            #forced to be posdef
            has_made_posdef_covar = cfmin.hasMadePosDefCovar(),
            hesse_failed = cfmin.hesseFailed(),
            has_covariance = cfmin.hasCovariance(),
            is_above_max_edm = cfmin.isAboveMaxEdm(),
            has_reached_call_limit = cfmin.hasReachedCallLimit()
        )
    return cfmin_struct


cdef minuitparam2struct(MinuitParameter mp):
    ret = Struct(
            number = mp.number(),
            name = mp.name(),
            value = mp.value(),
            error = mp.error(),
            is_const = mp.isConst(),
            is_fixed = mp.isFixed(),
            has_limits = mp.hasLimits(),
            has_lower_limit = mp.hasLowerLimit(),
            has_upper_limit = mp.hasUpperLimit(),
            lower_limit = mp.lowerLimit(),
            upper_limit = mp.upperLimit(),
        )
    return ret


cdef minoserror2struct(MinosError m):
    ret = Struct(
        lower = m.lower(),
        upper = m.upper(),
        is_valid = m.isValid(),
        lower_valid = m.lowerValid(),
        upper_valid = m.upperValid(),
        at_lower_limit = m.atLowerLimit(),
        at_upper_limit = m.atUpperLimit(),
        at_lower_max_fcn = m.atLowerMaxFcn(),
        at_upper_max_fcn = m.atUpperMaxFcn(),
        lower_new_min = m.lowerNewMin(),
        upper_new_min = m.upperNewMin(),
        nfcn = m.nfcn(),
        min = m.min()
        )
    return ret
