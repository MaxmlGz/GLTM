"""
Test function to demonstrate functionality.

This is python 2.7.
"""

from numpy import log, ascontiguousarray, ones, float32
from numpy.random import randint
from array import array

import cppyy
cppyy.include("foo.h")
cppyy.load_library("foo")
from cppyy.gbl import Foo
# Class stored here
f = Foo()


def baseline_tests():
    x = 27
    print "Testing the logarithm for x = {}:\n".format(x)
    print "When x = {}, then log(x) = {}\n".format(x, f.logFun(x))
    print "Answer from numpy, log(x) = {}\n".format(log(x))


def array_tests():
    X = ones((2, 3))
    X = randint(5, size=(2, 3))
    print "This is the matrix in python : {}\n".format(X)

    n, d = X.shape
    Xin = ascontiguousarray(X)
    f.arrayFuncs(Xin, n, d)

def multi_dim_array_test():
    W = [[1,2],[3,4]]
    Win = ascontiguousarray(W, dtype=float32)
    print f.multidimensionalArrays(Win)


if __name__ == '__main__':
    # baseline_tests()
    # array_tests()
    multi_dim_array_test()
