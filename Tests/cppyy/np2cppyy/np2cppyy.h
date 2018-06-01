#include <gsl/gsl_matrix.h>

// Main class which contains all the wrappers.
class np2cppyy
{
  public:
    int twoDimArrayTest(gsl_matrix *X, int N, int D);
    int oneDimArrayTest(double* X, int D);
};