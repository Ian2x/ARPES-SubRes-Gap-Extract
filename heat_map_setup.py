from general_functions_and_constants import *

##################################################
# DISPERSION MAIN FUNCTIONS
##################################################

# Normal Dispersion
def e(k, a, c):
    return a * k ** 2 + c

# Superconducting Dispersion
def E(k, a, c, dk):
    return (e(k, a, c) ** 2 + dk ** 2) ** 0.5


# Coherence Factors (relative intensity of BQP bands above and below EF)
def u(k, a, c, dk):
    if (dk == 0):
        if (a * k ** 2 + c > 0):
            return 1
        elif (a * k ** 2 + c < 0):
            return 0
        else:
            return 0.5
    return 0.5 * (1 + e(k, a, c) / E(k, a, c, dk))


u_vectorized = np.vectorize(u)


def v(k, a, c, dk):
    return 1 - u_vectorized(k, a, c, dk)


# BCS Spectral Function (https://arxiv.org/pdf/cond-mat/0304505.pdf) (non-constant gap)
def A_BCS(k, w, a, c, dk, T):
    return (1 / math.pi) * (
            u_vectorized(k, a, c, dk) * T / ((w - E(k, a, c, dk)) ** 2 + T ** 2) + v(k, a, c, dk) * T / (
            (w + E(k, a, c, dk)) ** 2 + T ** 2))


# Intensity Pre-factor
def Io(k):
    return 1;


# Full Composition Function (Knows a, c, dk, and T)
def Io_n_A_BCS(k, w, a, c, dk, T):
    return Io(k) * n_vectorized(w) * A_BCS(k, w, a, c, dk, T)

# Intensity
def I(k, w, a, c, dk, T, scaleup_factor, energy_conv_sigma):
    return energy_convolution_map(k, w, a, c, dk, T, Io_n_A_BCS, R_vectorized, scaleup_factor, energy_conv_sigma)
