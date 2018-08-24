from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from iminuit.color import Gradient

__all__ = ['LatexTable']


class LatexTable:
    """Latex table output.
    """
    float_format = '%10.5g'
    int_format = '%d'
    latex_kwd = [
        'alpha', 'beta', 'gamma',
        'delta', 'epsilon', 'zeta',
        'eta', 'theta', 'iota',
        'kappa', 'lambda', 'mu',
        'nu', 'xi', 'omicron',
        'pi', 'rho', 'sigma',
        'tau', 'upsilon', 'phi',
        'chi', 'psi', 'omega',
        'Alpha', 'Beta', 'Gamma',
        'Delta', 'Epsilon', 'Zeta',
        'Eta', 'Theta', 'Iota',
        'Kappa', 'Lambda', 'Mu',
        'Nu', 'Xi', 'Omicron',
        'Pi', 'Rho', 'Sigma',
        'Tau', 'Upsilon', 'Phi',
        'Chi', 'Psi', 'Omega',
    ]

    def __init__(self, data, headers=None, smart_latex=True,
                 escape_under_score=True, alignment=None, rotate_header=False,
                 latex_map=None):
        # Make sure #data columns matches #header columns, using any non-zero
        # length if data or headers are missing
        if len(data) > 0:
            num_col = len(data[0])
            if headers:
                assert num_col == len(headers)
        else:
            if headers is not None:
                num_col = len(headers)
            else:
                # LaTeX requires at least one column
                num_col = 1

        self.headers = headers
        self.data = data
        self.num_col = num_col
        self.smart_latex = smart_latex
        self.escape_under_score = escape_under_score
        self.alignment = self._auto_align() if alignment is None else alignment
        self.rotate_header = rotate_header
        self.latex_map = {} if latex_map is None else latex_map
        self.cell_color = {}  # map of tuple (i,j)=>(r, g, b) #i,j include header

    def _auto_align(self):
        return '|' + 'c|' * self.num_col

    def _format(self, s):
        if s in self.latex_map:
            return self.latex_map[s]
        elif isinstance(s, float):
            return self.float_format % s
        elif isinstance(s, int):
            return self.int_format % s
        elif self.smart_latex:
            return self._convert_smart_latex(s)
        elif self.escape_under_score:
            return s.replace('_', r'\_')
        else:
            return s

    def _convert_smart_latex(self, s):
        """convert greek symbol to latex one
        transform
        a to $a$ if a is greek letter else just a
        a_xxx to $a_{xxx}$ and
        a_xxx_yyy_zzz to a xxx $yyy_{zzz}$
        """
        # FIXME: implement this

        parts = s.split('_')
        if len(parts) == 1:  # a to $a$ if a is greek letter else just a
            if parts[0] in self.latex_kwd:
                return r'$\%s$' % str(parts[0])
            else:
                return str(parts[0])
        elif len(parts) == 2:  # a_xxx to $a_{xxx}$ and
            first = '\\%s' % parts[0] if parts[0] in self.latex_kwd else parts[0]
            second = '\\%s' % parts[1] if parts[1] in self.latex_kwd else parts[1]
            return r'$%s_{%s}$' % (first, second)
        else:  # a_xxx_yyy_zzz to a xxx $yyy_{zzz}$
            textpart = map(self._convert_smart_latex, parts[:-2])
            textpart = ' '.join(textpart)
            latexpart = self._convert_smart_latex('_'.join(parts[-2:]))
            return textpart + ' ' + latexpart

    def set_cell_color(self, i, j, c):
        """colorize i,j cell with rgb color tuple c

        Note that i,j index includes header.
        i=0 is header if header is present. If header is not present then
        i=0 refer to first data row.
        """
        self.cell_color[(i, j)] = c

    def _prepare(self):  # return list of list
        ret = []
        if self.headers:
            tmp = list(map(self._format, self.headers))
            if self.rotate_header:
                tmp = list(map(lambda x: '\\rotatebox{90}{%s}' % x, tmp))

            ret.append(tmp)
        for x in self.data:
            ret.append(list(map(self._format, x)))
        return ret

    def __str__(self):
        hline = '\\hline\n'
        ret = ''
        if len(self.cell_color) != 0:
            ret += '%\\usepackage[table]{xcolor} % include this for color\n'
            ret += '%\\usepackage{rotating} % include this for rotate header\n'
            ret += '%\\documentclass[xcolor=table]{beamer} % for beamer\n'
        ret += '\\begin{tabular}{%s}\n' % self.alignment
        ret += hline
        tdata = self._prepare()
        # decorate it

        for (i, j), c in self.cell_color.items():
            xcolor = '[RGB]{%d,%d,%d}' % (c[0], c[1], c[2])
            tdata[i][j] = '\\cellcolor' + xcolor + ' ' + tdata[i][j]

        for line in tdata:
            ret += ' & '.join(line) + '\\\\\n'
            ret += hline
        ret += '\\end{tabular}\n'

        return ret.strip()


class LatexFactory:
    @classmethod
    def build_matrix(cls, vnames, matrix, latex_map=None):
        """build latex correlation matrix"""
        # ret_link  = '<a onclick="$(\'#%s\').toggle()" href="#">Show Latex</a>'%uid
        headers = [''] + list(vnames)
        data = []
        color = {}
        for i, v1 in enumerate(vnames):
            tmp = [v1]
            for j, v2 in enumerate(vnames):
                m = matrix[i][j]
                tmp.append(m)
                color[(i + 1, j + 1)] = Gradient.color_for(abs(m))
                # +1 for header on the side and top
            data.append(tmp)

        table = LatexTable(headers=headers, data=data, rotate_header=True,
                           latex_map=latex_map)
        table.float_format = '%3.2f'
        for (i, j), c in color.items():
            table.set_cell_color(i, j, c)
        return table

    @classmethod
    def build_param_table(cls, mps, merr=None, float_format='%5.3e',
                          smart_latex=True, latex_map=None):
        """build latex parameter table"""
        headers = ['', 'Name', 'Value', 'Hesse Error', 'Minos Error-',
                   'Minos Error+', 'Limit-', 'Limit+', 'Fixed?', ]

        data = []
        for i, mp in enumerate(mps):
            minos_p, minos_m = ('', '') if merr is None or mp.name not in merr else \
                ('%g' % merr[mp.name].upper, '%g' % merr[mp.name].lower)
            limit_p = '' if mp.upper_limit is None else '%g' % mp.upper_limit
            limit_m = '' if mp.lower_limit is None else '%s' % mp.lower_limit
            fixed = 'Yes' if mp.is_fixed else 'No'
            tmp = [
                i,
                mp.name,
                '%g' % mp.value,
                '%g' % mp.error,
                minos_m,
                minos_p,
                limit_m,
                limit_p,
                fixed,
            ]
            data.append(tmp)
        alignment = '|c|r|r|r|r|r|r|r|c|'
        ret = LatexTable(data, headers=headers, alignment=alignment,
                         smart_latex=smart_latex, latex_map=latex_map)
        ret.float_format = float_format
        return ret
