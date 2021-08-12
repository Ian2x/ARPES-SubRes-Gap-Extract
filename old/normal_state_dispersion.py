import math
from general_functions_and_constants import *
from heat_map_setup import *
# build normal state dispersion map
thickness_sigma = 2
def A_e(k,w):
    dw = math.fabs(e(k)-w)
    return R_vectorized(dw, thickness_sigma)
A_e_vectorized = np.vectorize(A_e)
def n_A_e(k,w):
    return n_vectorized(w)*A_e_vectorized(k,w)
n_A_e_vectorized = np.vectorize(n_A_e)
def n_I(k,w):
    return energy_convolution_map(k,w,n_A_e,R_vectorized)