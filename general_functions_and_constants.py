from control_panel import *

import numpy as np
import math

##################################################
# GENERAL FUNCTIONS
##################################################

# Constants
hbar = 6.582 * 10 ** (-13)  # meV*s
mass_electron = 5.1100 * 10 ** 8  # mev/c^2
speed_light = 2.998 * 10 ** 8  # m/s


# Gaussian Function
def R(dw, sigma):
    # For in-exact approximation:
    # if (0.5 * (dw) ** 2 / sigma ** 2 > 100): return 0
    return (1 / sigma / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (dw) ** 2 / sigma ** 2)


R_vectorized = np.vectorize(R, excluded=['sigma'])


# Energy Convolution to a Map
def energy_convolution_map(k, w, main_func_k_w, conv_func_w, scaleup=scaleup_factor):
    # k should be 2d
    height = math.floor(k.size / k[0].size)
    width = k[0].size

    results = np.zeros((height, width))

    # === to test w/out convolution === #
    '''
    for i in range(height):
        for j in range(width):
            results[i][j] = scaleup * func_k_w(k[i][j], w[i][j])
    return results
    '''
    # === to test w/out convolution === #

    # Extract vertical arrays from 2d array --> to convolve over w
    inv_k = np.array([list(i) for i in zip(*k)])
    inv_w = np.array([list(i) for i in zip(*w)])
    # Flip vertically
    rev_inv_w = np.flip(inv_w)

    for i in range(height):
        for j in range(width):
            # 1D array of w at point (i, j) (in numpy coordinates); w is used to find dw
            curr_w = np.full(inv_w[j].size, w[i][j])
            # Energy convolution to find final intensity at point (i, j)
            res = np.convolve(scaleup * main_func_k_w(inv_k[j], inv_w[j]),
                              conv_func_w(rev_inv_w[j] - curr_w, energy_conv_sigma), mode='valid')
            results[i][j] = res

    return results


# Energy Convolution to an array
def energy_convolution(w, main_func, conv_func):
    results = np.zeros(w.size)

    # Flip vertically
    rev_w = np.flip(w)

    #
    for i in range(w.size):
        curr_w = np.full(w.size, w[i])
        res = np.convolve(main_func(w), conv_func(rev_w - curr_w), mode='valid')
        results[i] = res

    return results


# Add noise
def add_noise(map):
    height = math.floor(map.size / map[0].size)
    width = map[0].size
    for i in range(height):
        for j in range(width):
            map[i][j] = np.random.poisson(map[i][j])
            '''
            if map[i][j] < 1:
                map[i][j] = 1
            else:
                map[i][j] = np.random.poisson(map[i][j])
                if map[i][j] == 0:
                    map[i][j] = 1

            # ensure no 0's for error sigma (for fitting)
            if map[i][j] < 1:
                map[i][j] = 1
            '''
    return


# Fermi-Dirac Function
def n(w, uP=0, temp=60):
    # Boltzmann's constant (meV/K)
    kB = 8.617 * 10 ** (-2)
    # h-bar: 6.582 * 10 ** (-13) (mev*s) # (Implicit bc already expressing energy)
    # if w > 150: return 0 # Rounding approximation
    # if w < -150: return 1
    return 1 / (math.exp((w - uP) / kB / temp) + 1)


n_vectorized = np.vectorize(n)


# Reduced-Chi Calculation
def manualRedChi(data, fit, absSigmaSquared, DOF=1):
    res = 0
    for i in range(data.size):
        res += (data[i] - fit[i]) ** 2 / absSigmaSquared[i]
    return res / DOF


# F-Test Calculation
def manualFTest(data, fit1, para1, fit2, para2, absSigmaSquared, n):
    # fit1 should be 'nested' within fit2
    if (para2 <= para1):
        return ValueError
    chi1 = manualRedChi(data, fit1, absSigmaSquared)
    chi2 = manualRedChi(data, fit2, absSigmaSquared)
    return ((chi1 - chi2) / (para2 - para1)) / (chi2 / (n - para2))


# Gaussian Function (Normalized)
def gaussian_form_normalized(x, sigma, mu):
    return 1 / (sigma * (2 * math.pi) ** 0.5) * math.e ** (-0.5 * ((x - mu) / sigma) ** 2)


# Gaussian Function (General)
def gaussian_form(x, a, b, c):
    return a * math.e ** ((- (x - b) ** 2) / (2 * c ** 2))


# Lorentz Function
def lorentz_form(x, a, b, c):
    return a * c / ((x - b) ** 2 + c ** 2)


# Parabola (No Horizontal Shift)
def no_shift_parabola_form(x, a, c):
    return a * x ** 2 + c
