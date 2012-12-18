__all__ = ['HtmlFrontend']
from IPython.core.display import display, HTML, display_html
from util import Struct
class Gradient:
    #from http://code.activestate.com/recipes/266466-html-colors-tofrom-rgb-tuples/
    @classmethod
    def color_for(cls, v, min=0., max=1., startcolor=(163,254,186),
                  stopcolor=(255,117,117)):
        c = [0]*3
        for i,sc in enumerate(startcolor):
            c[i] = round(startcolor[i] + \
                   1.0*(v-min)*(stopcolor[i]-startcolor[i])/(max-min))
        return tuple(c)

    @classmethod
    def rgb_color_for(cls,v):
        c = cls.color_for(abs(v))
        return 'rgb(%d,%d,%d)'%c

good_style = 'background-color:#92CCA6'
bad_style = 'background-color:#FF7878'
warn_style = 'background-color:#FFF79A'

def good(x,should_be):
    return good_style if x == should_be else bad_style

def caution(x,should_be):
    return good_style if x == should_be else warn_style

def fmin_style(sfmin):
    """convert sfmin to style"""
    return Struct(
        is_valid=good(sfmin.is_valid,True),
        has_valid_parameters=good(sfmin.has_valid_parameters,True),
        has_accurate_covar=good(sfmin.has_accurate_covar,True),
        has_posdef_covar=good(sfmin.has_posdef_covar,True),
        has_made_posdef_covar=good(sfmin.has_made_posdef_covar,False),
        hesse_failed=good(sfmin.hesse_failed,False),
        has_covariance=good(sfmin.has_covariance,True),
        is_above_max_edm=good(sfmin.is_above_max_edm,False),
        has_reached_call_limit=caution(sfmin.has_reached_call_limit,False),
        )


def minos_style(smerr):
    """Convert minos error to style"""
    return Struct(
        is_valid = good(smerr.is_valid,True),
        lower_valid = good(smerr.lower_valid,True),
        upper_valid = good(smerr.upper_valid,True),
        at_lower_limit = good(smerr.at_lower_limit,False),
        at_upper_limit = good(smerr.at_upper_limit,False),
        at_lower_max_fcn = good(smerr.at_lower_max_fcn,False),
        at_upper_max_fcn = good(smerr.at_upper_max_fcn,False),
        lower_new_min = good(smerr.lower_new_min,False),
        upper_new_min = good(smerr.upper_new_min,False),
        )


class HtmlFrontend:

    def print_fmin(self, sfmin, tolerance=None, ncalls=0):
        """Display FunctionMinum in html representation. 

        .. note: Would appreciate if someone would make jquery hover 
        description for each item."""
        goaledm = 0.0001*tolerance*sfmin.up
        style = fmin_style(sfmin)
        header = u"""
        <table>
            <tr>
                <td>FCN = {sfmin.fval}</td>
                <td>NFCN = {sfmin.nfcn}</td>
                <td>NCALLS = {ncalls}</td>
            </tr>
            <tr>
                <td>EDM = {sfmin.edm}</td>
                <td>GOAL EDM = {goaledm}</td>
                <td>UP = {sfmin.up}</td>
            </tr>
        </table>
        """.format(**locals())
        status = u"""
        <table>
            <tr>
                <td align="center">Valid</td>
                <td align="center">Valid Param</td>
                <td align="center">Accurate Covar</td>
                <td align="center">PosDef</td>
                <td align="center">Made PosDef</td>
            </tr>
            <tr>
                <td align="center" style="{style.is_valid}">{sfmin.is_valid!r}</td>
                <td align="center" style="{style.has_valid_parameters}">{sfmin.has_valid_parameters!r}</td>
                <td align="center" style="{style.has_accurate_covar}">{sfmin.has_accurate_covar!r}</td>
                <td align="center" style="{style.has_posdef_covar}">{sfmin.has_posdef_covar!r}</td>
                <td align="center" style="{style.has_made_posdef_covar}">{sfmin.has_made_posdef_covar!r}</td>
            </tr>
            <tr>
                <td align="center">Hesse Fail</td>
                <td align="center">HasCov</td>
                <td align="center">Above EDM</td>
                <td align="center"></td>
                <td align="center">Reach calllim</td>
            </tr>
            <tr>
                <td align="center" style="{style.hesse_failed}">{sfmin.hesse_failed!r}</td>
                <td align="center" style="{style.has_covariance}">{sfmin.has_covariance!r}</td>
                <td align="center" style="{style.is_above_max_edm}">{sfmin.is_above_max_edm!r}</td>
                <td align="center"></td>
                <td align="center" style="{style.has_reached_call_limit}">{sfmin.has_reached_call_limit!r}</td>
            </tr>
        </table>
        """.format(**locals())
        display_html(header+status,raw=True)


    def print_merror(self, vname, smerr):
        stat = 'VALID' if smerr.is_valid else 'PROBLEM'
        style = minos_style(smerr)
        to_print = """
        <span>Minos status for {vname}: <span style="{style.is_valid}">{stat}</span></span>
        <table>
            <tr>
                <td>Error</td>
                <td>{smerr.lower}</td>
                <td>{smerr.upper}</td>
            </tr>
            <tr>
                <td>Valid</td>
                <td style="{style.lower_valid}">{smerr.lower_valid}</td>
                <td style="{style.upper_valid}">{smerr.upper_valid}</td>
            </tr>
            <tr>
                <td>At Limit</td>
                <td style="{style.at_lower_limit}">{smerr.at_lower_limit}</td>
                <td style="{style.at_upper_limit}">{smerr.at_upper_limit}</td>
            </tr>
            <tr>
                <td>Max FCN</td>
                <td style="{style.at_lower_max_fcn}">{smerr.at_lower_max_fcn}</td>
                <td style="{style.at_upper_max_fcn}">{smerr.at_upper_max_fcn}</td>
            </tr>
            <tr>
                <td>New Min</td>
                <td style="{style.lower_new_min}">{smerr.lower_new_min}</td>
                <td style="{style.upper_new_min}">{smerr.upper_new_min}</td>
            </tr>
        </table>
        """.format(**locals())
        display_html(to_print, raw=True)


    def print_param(self, mps, merr=None):
        """print list of parameters"""
        #todo: how to make it right clickable to export to latex
        to_print = ""
        header = """
        <table>
            <tr>
                <td></td>
                <td>Name</td>
                <td>Value</td>
                <td>Parab Error</td>
                <td>Minos Error-</td>
                <td>Minos Error+</td>
                <td>Limit-</td>
                <td>Limit+</td>
                <td>FIXED</td>
            </tr>
        """.format(**locals())
        to_print += header
        for i,mp in enumerate(mps):
            minos_p, minos_m = (0.,0.) if merr is None or mp.name not in merr else\
                               (merr[mp.name].upper, merr[mp.name].lower)
            limit_p, limit_m = ('','') if not mp.has_limits else\
                               (mp.upper_limit, mp.lower_limit)
            fixed = 'FIXED' if mp.is_fixed else ''
            j = i+1
            content = """
            <tr>
                <td>{j}</td>
                <td>{mp.name}</td>
                <td>{mp.value:e}</td>
                <td>{mp.error:e}</td>
                <td>{minos_m:e}</td>
                <td>{minos_p:e}</td>
                <td>{limit_m!s}</td>
                <td>{limit_p!s}</td>
                <td>{fixed}</td>
            </tr>
            """.format(**locals())
            to_print += content
        to_print += """
            </table>
        """
        display_html(to_print, raw=True)


    def print_banner(self, cmd):
        #display_html('<h2>%s</h2>'%cmd, raw=True)
        pass


    def print_matrix(self, vnames, matrix, varno=None):
        to_print = """
            <table>
                <tr>
                    <td></td>
        """
        for v in vnames:
            to_print += """
            <td>
            <span style="-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
            {v}
            </span>
            </td>
            """.format(**locals())
        to_print += """
                </tr>
                """
        for i,v1 in enumerate(vnames):
            to_print += """
            <tr>
                <td>{v1}</td>
            """.format(**locals())
            for j,v2 in enumerate(vnames):
                val = matrix[i][j]
                color = Gradient.rgb_color_for(val)
                to_print += """
                <td style="background-color:{color}">
                {val:3.2f}
                </td>
                """.format(**locals())
            to_print += """
            </tr>
            """
        to_print += '</table>'
        display_html(to_print, raw=True)


    def print_hline(self):
        display_html('<hr>', raw=True)


class HTMLWrapper:
    def __init__(self, txt):
        self.data = txt
    def _repr_html_(self):
        return txt
