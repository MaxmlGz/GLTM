// My header
#include "foo.h"

// Standard
#include <iostream>

// GSL headers
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h> // GSL_NEGINF lives here

int Foo::bar(void)
{
    double x = 15.0;
    std::cout << std::fixed;
    double y = gsl_sf_bessel_J0(x);
    std::cout << y;
    return 0;
}

// Log function depending on other gsl
double Foo::logFun(double x)
{
    if (x == 0)
    {
        return GSL_NEGINF;
    }
    else if (x < 0)
    {
        printf("Error: logarithm is not defined for negative numbers\n");
        return GSL_NEGINF;
    }
    else
    {
        return gsl_sf_log(x);
    }
}

int Foo::arrayFuncs(gsl_matrix *X)
{
    // Get array shape
    int N = X->size1;
    int D = X->size2;
    return N, D;
}

// Testing array behaviour
double Foo::printArrayMembers(gsl_matrix *W)
{
    printf("This is some instances of elements:\nW[0][0]=%f, W[0][1]=%f, W[1][0]=%f\n\n", gsl_matrix_get(W, 0, 0), gsl_matrix_get(W, 0, 1), gsl_matrix_get(W, 1, 0));
    // Get array shape
    int N = W->size1;
    int D = W->size2;
    printf("Matrix shape = (N:%i, D:%i)", N, D);

    // Loop over columns
    for (int d = 0; d < D; d++)
    {
        double wnd;
        // Loop over rows
        for (int n = 0; n < N; n++)
        {
            // Get indexed element
            wnd = gsl_matrix_get(W, n, d);
            printf("This is element w[%d,%d] = %f\n", n, d, wnd);
        }
    }
}

outputs Foo::multiOutputTest()
{
    outputs output;
    output.Kest = 10;
    output.countErr = 5;
    // Test matrix allocation
    gsl_matrix *A = gsl_matrix_calloc(2, 2);
    gsl_matrix_set(A, 0, 0, 1);
    gsl_matrix_set(A, 0, 1, 2);
    gsl_matrix_set(A, 1, 0, 3);
    gsl_matrix_set(A, 1, 1, 4);
    // Assign to struct
    output.West = *A;

    return output;
    gsl_matrix_free(A);
}
