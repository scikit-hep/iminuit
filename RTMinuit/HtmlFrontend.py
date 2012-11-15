__all__ = ['HtmlFrontend']
from IPython.core.display import display, HTML, display_html
class Gradient:
    #A3FEBA pastel green
    #FF7575 pastel red
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


class HtmlFrontend:

    def print_fmin(self, sfmin, tolerance=None, ncalls=0):
        """Display FunctionMinum in html representation"""
        goaledm = 0.0001*tolerance*sfmin.up
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
                <td align="center">{sfmin.is_valid!r}</td>
                <td align="center">{sfmin.has_valid_parameters!r}</td>
                <td align="center">{sfmin.has_accurate_covar!r}</td>
                <td align="center">{sfmin.has_posdef_covar!r}</td>
                <td align="center">{sfmin.has_made_posdef_covar!r}</td>
            </tr>
            <tr>
                <td align="center">Hesse Fail</td>
                <td align="center">HasCov</td>
                <td align="center">Above EDM</td>
                <td align="center"></td>
                <td align="center">Reach calllim</td>
            </tr>
            <tr>
                <td align="center">{sfmin.hesse_failed!r}</td>
                <td align="center">{sfmin.has_covariance!r}</td>
                <td align="center">{sfmin.is_above_max_edm!r}</td>
                <td align="center"></td>
                <td align="center">{sfmin.has_reached_call_limit!r}</td>
            </tr>
        </table>
        """.format(**locals())
        display_html(header+status,raw=True)


    def print_merror(self, vname, smerr):
        stat = 'VALID' if smerr.is_valid else 'PROBLEM'

        to_print = """
        <span>Minos status for {vname}: {stat}<span>
        <table>
            <tr>
                <td>Error</td>
                <td>{smerr.lower}</td>
                <td>{smerr.upper}</td>
            </tr>
            <tr>
                <td>Valid</td>
                <td>{smerr.lower_valid}</td>
                <td>{smerr.upper_valid}</td>
            </tr>
            <tr>
                <td>At Limit</td>
                <td>{smerr.at_lower_limit}</td>
                <td>{smerr.at_upper_limit}</td>
            </tr>
            <tr>
                <td>Max FCN</td>
                <td>{smerr.at_lower_max_fcn}</td>
                <td>{smerr.at_upper_max_fcn}</td>
            </tr>
            <tr>
                <td>New Min</td>
                <td>{smerr.lower_new_min}</td>
                <td>{smerr.upper_new_min}</td>
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
                <td></td>
            </tr>
        """.format(**locals())
        to_print += header
        for i,mp in enumerate(mps):
            minos_p, minos_m = (0.,0.) if merr is None or mp.name not in merr else\
                               (merr[mp.name].upper, merr[mp.name].lower)
            limit_p, limit_m = ('','') if mp.has_limits else\
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
        display_html('<h2>%s</h2>'%cmd, raw=True)


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
