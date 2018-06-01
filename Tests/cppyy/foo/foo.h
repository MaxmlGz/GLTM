#include <gsl/gsl_matrix.h>
#include <iostream>
#include <stdio.h>

struct outputs 
{
    // Results structure (shadow the matlab script calls)
    std::int64_t Kest; // K estimate (this is parametric and constant throughout)
    std::int64_t countErr; // Count errors
    gsl_matrix West; // Likelihood model weights
    double LIK; // Observation likelihood
};

class Foo
{
public:
  int bar();
  double logFun(double x);
  double printArrayMembers(gsl_matrix *W);
  outputs multiOutputTest();
  int arrayFuncs(gsl_matrix *X);
};















// Data strctures to take care of [Valera's notation]:
// C++ side of execution
// --------------------------
// 2D arrays: X, paramW,
// 1D array: C, R,
// Floats: s2Z, s2B, s2Y, s2u, s2theta, Nsim, maxK, XT
