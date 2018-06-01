"""
Testing of GSL routines for python 3.
"""

# Wim's decorator lives here
from gsl_decorators import *

import cppyy
cppyy.load_library("/usr/lib/libgsl")  # Change this to your local setting.
cppyy.include("gsl/gsl_matrix.h")
cppyy.include("foo.h")
cppyy.load_library("foo")
# This raises an error for some reason, but runs anyway.
from cppyy.gbl import Foo

# Class stored here
gsl_test = Foo()


def assign_gsl_data_standard(X):
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
    N, D = X.shape
    gm = cppyy.gbl.gsl_matrix()
    gm.size1 = N
    gm.size2 = D
    gm.tda = D
    gm.data = X.flatten().astype('float64')
    gm.owner = 0

    return gm


def run_gsl_print_routine():
    """
    This function creates some dummy data, passes it to the cppyy wrapeed C++ function
    which prints out the data.
    """
    # Random non-symmetric matrix
    X = np.random.randint(8, size=(3, 4))
    # Convert to GSL format, using Wim's converter
    # Wim's decorator
    # X_gsl = numpy2gsl(X)
    # Use the standard method
    X_gsl = assign_gsl_data_standard(X)

    # Pass to wrapped routine, which should print out X.
    print("Enter the dragon (C++ wrapper)...\n")
    gsl_test.printArrayMembers(X_gsl)


if __name__ == '__main__':
    # run_gsl_print_routine()
    out = gsl_test.multiOutputTest()
    print(out.K)
    print(out.countErr)
    print(out.weights)
    print(type(out.weights))
