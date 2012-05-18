
class MinuitHTMLResult:
    def __init__(self,m):
        self.values = m.values
        self.errors = m.errors
    def _repr_html_(self):
        tmp = []
        header = u'<tr><td>Parameter</td><td>Value</td><td>Error</td></tr>'
        keys = sorted(self.values)
        for k in keys:
            line = u'<tr><td align="left">%s</td><td align="right">%e</td><td align="left"> &plusmn;%e</td></tr>'%(k,self.values[k],self.errors[k])
            tmp.append(line)
        ret =  '<table>%s\n%s\n</table>'%(header,'\n'.join(tmp))
        return ret

class MinuitCorrelationMatrixHTML:
    def __init__(self,m):
        self.matrix = m.error_matrix(True)
        self.pos2var = m.pos2var
