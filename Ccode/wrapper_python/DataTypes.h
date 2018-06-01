#include <math.h>
#include <stdio.h>
#include <iostream>
#include <time.h>

#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_randist.h"

// To expose this class to Python.
// class GLTM
// {
//   public:
//     // Keeping commonality with the GLFM, we call this function infer.
//     void infer();
// };