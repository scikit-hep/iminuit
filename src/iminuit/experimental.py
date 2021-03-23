"""
Warning: As the name indicates, everything in this module is experimental.

The API is not final, code here may be removed or altered in breaking ways without
warning.

Use at your own risk.
"""

from .util import merge_signatures


def expanded(*callables):
    """
    Return expanded callables with unified signatures.

    Uses :func:`merge_signatures` to unify the signatures and generates function
    wrappers that accept the merged signature and call the original function in the
    correct way.

    This is best explained by an example::

        def f(x, y, z): ...

        def g(x, p): ...

        f2, g2 = expand_functions(f, g)

        # f2 is the equivalent of: def f2(x, y, z, p): return f(x, y, z)
        # g2 is the equivalent of: def g2(x, y, z, p): return g(x, p)
    """
    varnames, mapping = merge_signatures(callables)
    total = ",".join(varnames)
    args = [",".join(varnames[i] for i in map) for map in mapping]
    lambdas = ",".join(
        f"lambda {total} : funcs[{i}]({arg})" for (i, arg) in enumerate(args)
    )
    return eval(lambdas, {"funcs": callables})
