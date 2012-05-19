
class MinuitHTMLResult:
    def __init__(self,m):
        """
        :param m:
        :type m Minuit:
        """

        self.varnames = m.varname
        self.values = m.values
        self.errors = m.errors
    def _repr_html_(self):
        tmp = []
        header = u'<tr><td></td><td>Parameter</td><td>Value</td><td>Error</td></tr>'
        keys = sorted(self.values)
        for i,k in enumerate(self.varnames):
            val = self.values[k]
            err = self.errors[k]
            varno = i+1
            line = u"""<tr>
                    <td align="right">{varno:d}</td>
                    <td align="left">{k}</td>
                    <td align="right">{val:e}</td>
                    <td align="left"> &plusmn;{err:e}</td>
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
        if val>0.8:
            return 'font-weight:bold;'
        return ''

    def _repr_html_(self):
        header = ''
        for i in range(self.nparams):
            header+='<td style="text-align:center;">%s</td>\n'%self.params[i]
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
