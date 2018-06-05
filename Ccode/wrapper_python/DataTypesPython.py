# Import numpy -> wrapper
import numpy as np
from array import array
import scipy.io
import os
import warnings
from copy import deepcopy
import cppyy
import re

dataset_dir = os.path.join(os.path.dirname(__file__), "../..", "datasets/")

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

cppyy.load_library("/usr/lib/libgsl")  # Change this to your local setting.
cppyy.include("gsl/gsl_matrix.h")
cppyy.include("wrapper_python.h")
cppyy.load_library("libDataTypes")
from cppyy.gbl import wrapperPython

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
            s.write(str(data[i * tda + j]))
        s.write(']')
        if i != self.size1 - 1:
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
    return gmv.matrix  # safe: cppyy will set a proper life line


def synthetic_data():
    """
    Function creates some synthetic data, that we use for test. The data
    is on the columns.

    Example data is: discrete, discrete and continuous with option real.
    """
    return np.array([[1, 3, 0.4], [3, 4, -1], [3, 4, 0.1], [3, 4, 0.2]])


def weight_assign(list_of_types):
    """
    Assigns weights based on column type.
    4: discrete
    3: binary
    2: continuous
    1: continuous with option positive
    """
    weights = dict([(4, [100, 100, 100, 1e-6]),  # Categorical, ordinal, count
                    (3, [1e-6, 1e-6, 1e-6, 1e-6]),  # binary
                    (2, [100, 100, 1e-6, 1e-6]),  # real and interval
                    (1, [100, 100, 1e-6, 100])])  # real, interval, positive real

    return np.array(list(map(lambda d: weights[d], list_of_types)))


def numpy_2_gsl_matrix_struct(x):
    """
    Function assigns numpy array to a gsl matrix in the standard way.

    gsl_matrix structure:

    typedef struct
    {
    size_t size1;
    size_t size2;
    size_t tda;
    double * data;
    gsl_block * block;
    int owner;
    } gsl_matrix;
    """
    N, D = x.shape
    gm = cppyy.gbl.gsl_matrix()
    gm.size1 = N
    gm.size2 = D
    gm.tda = D
    gm.data = x.flatten().astype('float64')
    gm.owner = 0

    return gm


def load_test_input_params():
    """
    This function provides the necessary input parameters for 
    DataType.cpp.
    """
    X = synthetic_data().T  # Initial shape N x D --> D x N
    D, _ = X.shape
    # discrete, discrete, continuous with option real
    list_of_types = [4, 4, 1]
    number_of_categories_per_d = [2, 2, 1]
    assert len(list_of_types) == D
    W = weight_assign(list_of_types)
    s2Z, s2B, s2Y, s2u, s2theta = 1., 1., 1., 0.001, 1.
    Xmiss = X
    Nits, KK = 2, 6
    # Matrices
    X, Xmiss = numpy2gsl(X), numpy2gsl(Xmiss)
    W = numpy2gsl(W)
    # Use built in array to talk to C++
    # This should be past as an array of unsigned integers
    list_of_types = array('i', list_of_types)
    number_of_categories_per_d = array('i', number_of_categories_per_d)

    return [X, list_of_types, number_of_categories_per_d, W, s2Z, s2B, s2Y, s2u, s2theta, Nits, KK, Xmiss]


def pass_params_to_cppyy_verbose():
    """
    This function passes all inputs to C++ method "sampler_function()".
    """
    # This is a list recall
    # args = load_test_input_params()
    wrap = wrapperPython()
    print("Entering C++ routine...\n")
    # wrap.sampler_function(*args)

    X = synthetic_data()
    X_gsl = numpy2gsl(X.T)  # numpy_2_gsl_matrix_struct(X)
    list_of_types = array('i', [1, 2, 3])
    number_of_categories_per_d = array('i', [2, 2, 1])
    W = weight_assign(list_of_types)
    W_gsl = numpy2gsl(W)  # numpy_2_gsl_matrix_struct(X)
    s2Z, s2B, s2Y, s2u, s2theta = 1., 1., 1., 0.001, 1.
    Nits, KK = 2, 6
    XT_gsl = numpy2gsl(X.T)  # numpy_2_gsl_matrix_struct(X)

    return wrap.verbose_sampler_function(
        X_gsl, list_of_types, number_of_categories_per_d, W_gsl, s2Z, s2B, s2Y, s2u, s2theta, Nits, KK, XT_gsl)


def run_simulation(dataset="AbaloneC",
                   s2Z=1.,
                   s2B=1.,
                   s2Y=1.,
                   s2u=0.001,
                   s2theta=1.,
                   KK=5,
                   Nits=5):
    """
    This function initialises and runs the GLTM with the same datasets as Valera.
    """

    # Load the C++ wrapper
    wrap = wrapperPython()
    # Get dataset
    warnings.filterwarnings("ignore")  # loadmat does not like matlab files new than v7.3
    mat = scipy.io.loadmat(dataset_dir + dataset + ".mat")
    # Observations; the missing test-set is loaded here if we are doing that [remember to transpose]
    mat_missing = scipy.io.loadmat(dataset_dir + dataset + 'Miss.mat')  # Indices of missing data, so we can compare to
    #  Valera

    # Remember to transpose the dataset before passing to GSL
    X_deep = deepcopy(mat['X'].T)  # We want a copy not a view since X is mutable otherwise
    X = numpy2gsl(X_deep)
    np.put(mat['X'], mat_missing['miss'].tolist()[0], -1)
    # Reshape back and transpose, then apply GSL converter
    Xmiss = numpy2gsl(mat['X'].T)

    # List of available types
    types = array('i', mat['T'].tolist()[0])
    # Number of categories for discrete types
    categories = array('i', mat['R'].tolist()[0])
    # Assign weight priors based on the available types
    weights = numpy2gsl(weight_assign(types))

    # TODO: we need to create a method whereby the results of this model are stored in a dict indexed by K. When a new K
    # is instantiated, the weights are stored there, otherwise if K already exists, the weights are appended to it,
    # meaning that in the end we take the mean of all weights indexed by one K.
    # weight_dict[K] <-- weights
    # What do we say about exploratory power of K?

    print("Entering C++ routine...\n")
    sim_out = wrap.verbose_sampler_function(Xmiss, types, categories, weights, s2Z, s2B, s2Y, s2u, s2theta, Nits, KK,
                                            X)

    # Store results
    sim_result = dict()
    sim_result['latent_features'] = sim_out.Kest
    sim_result['countErr'] = sim_out.countErr
    sim_result['weights'] = np.asarray(sim_out.West)
    sim_result['likelihoods'] = np.asarray(sim_out.LIK)

    return sim_result, mat['T'].tolist()[0]


def pass_params_to_cppyy():
    """
    This function passes all inputs to C++ method "sampler_function()".
    """
    # This is a list recall
    args = load_test_input_params()
    wrap = wrapperPython()
    print("Entering C++ routine...\n")
    wrap.sampler_function(*args)


if __name__ == '__main__':
    # Verbose version
    out, types = run_simulation()
    print(out['weights'])
    print(out['likelihoods'])
    print("\n", types)
