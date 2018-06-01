
#include <gsl/gsl_matrix.h>
#include <iostream>
#include <stdio.h>

struct outputs 
{
    // Results structure (shadow the matlab script calls)
    std::int64_t Kest; // K estimate (this is parametric and constant throughout)
    std::int64_t countErr; // Count errors
    gsl_matrix West; // Likelihood model weights (recall that these are of type double)
    gsl_matrix LIK; // Observation likelihood (recall that these are of type double)
};

// Main class which contains the main simulation routine
class wrapperPython
{
public:
  // Working wrapper 
  outputs verbose_sampler_function(gsl_matrix *X, int *C, int *R, gsl_matrix *paramW, double s2Z, double s2B, double s2Y, double s2u, double s2theta, int Nsim, int maxK, gsl_matrix *XT);
  // OBS: currently does not work
  void sampler_function(gsl_matrix *X, int *C, int *R, gsl_matrix *paramW, double s2Z, double s2B, double s2Y, double s2u, double s2theta, int Nsim, int maxK, gsl_matrix *XT);
};