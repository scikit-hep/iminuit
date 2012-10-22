
class MinuitHTMLResult:
    def __init__(self,m):
        """
        :param m:
        :type m Minuit:
        """

        self.varnames = m.varname
        self.values = m.values
        self.errors = m.errors
        self.mnerrors = m.minos_errors()
    def _repr_html_(self):
        tmp = []
        header = u"""<tr>
            <td></td>
            <td>Parameter</td>
            <td>Value</td>
            <td>Parab Error</td>
            <td>Minos Error-</td>
            <td>Minos Error+</td>
            </tr>"""
        keys = sorted(self.values)
        for i,k in enumerate(self.varnames):
            val = self.values[k]
            err = self.errors[k]
            mnp = self.mnerrors[k].eplus
            mnm = self.mnerrors[k].eminus
            varno = i+1
            line = u"""<tr>
                    <td align="right">{varno:d}</td>
                    <td align="left">{k}</td>
                    <td align="right">{val:e}</td>
                    <td align="left"> &plusmn;{err:e}</td>
                    <td align="left">{mnm:e}</td>
                    <td align="left">+{mnp:e}</td>
                    </tr>""".format(**locals())
            tmp.append(line)
        ret =  '<table>%s\n%s\n</table>'%(header,'\n'.join(tmp))
        return ret

class MinuitCorrelationMatrixHTML:
    def __init__(self,m):
        self.matrix = m.error_matrix(True)
        self.params = m.list_of_vary_param()
        self.nparams = len(self.params)
        assert(self.matrix.shape==(self.nparams,self.nparams))

    def style(self,val):
        return 'background-color:%s'%Gradient.rgb_color_for(val)

    def _repr_html_(self):
        header = ''
        for i in range(self.nparams):
            header+='<td style="text-align:left"><div style="-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">%s</div></td>\n'%self.params[i]
        header = '<tr><td></td>\n%s</tr>\n'%header
        lines = list()
        for i in range(self.nparams):
            line = '<td>%s</td>'%self.params[i]
            for j in range(self.nparams):
                style = self.style(self.matrix[i,j])
                line+='<td style="%s">%4.2f</td>\n'%(style,self.matrix[i,j])
            line = '<tr>\n%s</tr>\n'%line
            lines.append(line)
        ret = '<table>\n%s%s</table>\n'%(header,''.join(lines))
        return ret

class Gradient:
    #A3FEBA pastel green
    #FF7575 pastel red
    #from http://code.activestate.com/recipes/266466-html-colors-tofrom-rgb-tuples/
    @classmethod
    def color_for(cls,v,min=0.,max=1.,startcolor=(163,254,186),stopcolor=(255,117,117)):
        c = [0]*3
        for i,sc in enumerate(startcolor):
            c[i] = round(startcolor[i] + 1.0*(v-min)*(stopcolor[i]-startcolor[i])/(max-min))
        return tuple(c)

    @classmethod
    def rgb_color_for(cls,v):
        c = cls.color_for(abs(v))
        return 'rgb(%d,%d,%d)'%c