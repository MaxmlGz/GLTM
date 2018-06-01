import numpy as np
import numpy as np
from array import array

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import cppyy
import re

# type matchers numpy -> gsl
tm_np2gsl = dict()
tm_np2gsl['float64'] = ''
tm_np2gsl['float32'] = 'float'
tm_np2gsl[cppyy.sizeof('long') == 4 and 'int32' or 'int64'] = 'long'
tm_np2gsl[cppyy.sizeof('int') == 4 and 'int32' or 'int64'] = 'int'
# etc. for other numpy types

converters = dict()
for key, value in tm_np2gsl.items():
    if key == 'float64':
        converters[key] = 'gsl_matrix_view_array'
    else:
        converters[key] = 'gsl_matrix_%s_view_array' % value

# gsl_matrix decorator


def gsl_matrix_repr(self):
    data, tda = self.data, self.tda
    s = StringIO()
    s.write('[')
    for i in range(self.size1):
        if i == 0:
            s.write('[')
        else:
            s.write(' [')
        for j in range(self.size2):
            if j != 0:
                s.write(', ')
            s.write(str(data[i*tda+j]))
        s.write(']')
        if i != self.size1-1:
            s.write('\n')
    s.write(']')
    return s.getvalue()


re_gsl_matrix = re.compile('gsl_matrix_?(\w*)')


def gsl_pythonizor(pyklass, pyname):
    res = re_gsl_matrix.match(pyname)
    if res and res.groups()[0] in tm_np2gsl.values():
        pyklass.__repr__ = gsl_matrix_repr


cppyy.py.add_pythonization(gsl_pythonizor)


def numpy2gsl(arr):
    try:
        converter = getattr(cppyy.gbl, converters[str(arr.dtype)])
    except KeyError as e:
        raise TypeError('unsupported data type: %s' % arr.dtype)
    from numpy import ascontiguousarray
    data = ascontiguousarray(arr)
    gmv = converter(data, arr.shape[0], arr.shape[1])
    gmv._keep_numpy_arr_alive = data
    return gmv.matrix     # safe: cppyy will set a proper life line
