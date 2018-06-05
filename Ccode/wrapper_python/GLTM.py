"""
Python wrapper methods for the C++ methods.
"""

from DataTypesPython import *
import time
from typing import Tuple

def infer(dataset="AbaloneC",
          s2Z=1.,
          s2B=1.,
          s2Y=1.,
          s2u=0.001,
          s2theta=1.,
          maxK=10,
          Niter=10) -> Tuple:
    """
    Python wrapper to launch inference routine for the GLTM model.
    Inputs:
        dataset (optional): input data N*D where:
                N = number of observations
                D = number of dimensions
        params (optional):
            s2Z:
            s2theta:
            s2Y: prior noise variance for pseudo-observations Y
            s2u: internal auxiliary noise
            s2B: noise variance for prior over elements of matrix B
            Niter: number of simulations
            maxK: maximum number of features for memory allocation
    Output:
        hidden:
            Z: feature activation matrix sampled from posterior
            B: observation matrix sampled from posterior
            s2Y: inferred noise variance for pseudo-observations Y
    """

    # Load the C++ wrapper
    wrap = wrapperPython()  # This is not actually an error just python being silly.

    # Get dataset
    warnings.filterwarnings("ignore")  # loadmat does not like matlab files newer than v7.3
    mat = scipy.io.loadmat(dataset_dir + dataset + ".mat")

    # Observations; the missing test-set is loaded here if we are doing that [remember to transpose]
    mat_missing = scipy.io.loadmat(dataset_dir + dataset + 'Miss.mat')

    # Remember to transpose the dataset before passing to GSL
    X_deep = deepcopy(mat['X'].T)  # We want a copy not a view since X is mutable otherwise

    # Transform into C-readable type
    X = numpy2gsl(X_deep)

    # Convert missing values to -1 (in place operation)
    np.put(mat['X'], mat_missing['miss'].tolist()[0], -1)

    # Reshape back and transpose, then apply GSL converter
    Xmiss = numpy2gsl(mat['X'].T)

    # List of available types (convert to C-readable type)
    types = array('i', mat['T'].tolist()[0])

    # Number of categories for discrete types
    categories = array('i', mat['R'].tolist()[0])

    # Assign weight priors based on the available types
    weights = numpy2gsl(weight_assign(types))

    print(time.ctime() + " -- Entering C++ routine on GLTM side...\n")
    sim_out = wrap.verbose_sampler_function(Xmiss, types, categories, weights, s2Z, s2B, s2Y, s2u, s2theta, Niter, maxK,
                                            X)
    print(time.ctime() + " -- Completed C++ routine on GLTM side...\n")

    # Store results
    sim_result = dict()
    sim_result['latent_features'] = sim_out.Kest  # This is a bit stupid to pass since it does not change.
    sim_result['countErr'] = sim_out.countErr
    sim_result['weights'] = np.asarray(sim_out.West)  # Convert back to numpy
    sim_result['likelihoods'] = np.asarray(sim_out.LIK)  # Convert back to numpy

    return sim_result, mat['T'].tolist()[0]
