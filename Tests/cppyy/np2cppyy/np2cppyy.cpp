
#include "np2cppyy.h"

// Standard
#include <iostream>
#include <stdio.h>

// GSL headers
#include <gsl/gsl_matrix.h>

// Get array shape and print matrix values from numpy matrix.
int np2cppyy::twoDimArrayTest(gsl_matrix *X, int N, int D)
{
    printf("This is printing inside the C++ function.\n");
    printf("N=%d, D=%d\n", N, D);
    printf("Printing part of test matrix...\n");
    printf("X[0,0]=%f, X[0,1]=%f, X[1,0]=%f\n", gsl_matrix_get(X, 0, 0), gsl_matrix_get(X, 0, 1), gsl_matrix_get(X, 1, 0));
}

int np2cppyy::oneDimArrayTest(double* X, int D)
{
    printf("This is printing inside the C++ function.\n");
    printf("N=%d, D=%d\n", D);
    printf("Printing part of test matrix...\n");
    printf("X[0]=%f, X[1]=%f, \n", X[0], X[1]);
}