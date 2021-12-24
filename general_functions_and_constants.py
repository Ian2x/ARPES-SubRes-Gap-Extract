from control_panel import *

import numpy as np
import math

##################################################
# GENERAL FUNCTIONS AND CONSTANTS
##################################################

# Constants
hbar = 6.582 * 10 ** (-13)  # meV*s
mass_electron = 5.1100 * 10 ** 8  # mev/c^2
speed_light = 2.998 * 10 ** 8  # m/s

ONE_BILLION = 1000000000

# Gaussian Function
def R(dw, sigma):
    '''
    Gaussian distribution for energy convolution
    :param dw: distance on energy axis between energy convolution point and nearby point
    :param sigma: energy resolution
    :return: a float representing convolution factor
    '''
    # For in-exact approximation:
    # if (0.5 * (dw) ** 2 / sigma ** 2 > 100): return 0
    return (1 / sigma / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (dw) ** 2 / sigma ** 2)


R_vectorized = np.vectorize(R, excluded=['sigma'])


def energy_convolution_map(k, w, main_func_k_w, conv_func_w, scaleup=scaleup_factor):
    '''
    Applies energy convolution to a map
    :param k: array of momentum
    :param w: array of energy
    :param main_func_k_w: function (k,w) representing map
    :param conv_func_w: function to convolute with (typically a Gaussian)
    :param scaleup: factor to scale-up resulting map by
    :return: convoluted map
    '''
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


def energy_convolution(w, main_func, conv_func):
    '''
    Energy Convolution to an array
    :param w: energy array
    :param main_func: function (w) representing array
    :param conv_func: function to convolute with (typically a Gaussian)
    :return: convoluted array
    '''
    results = np.zeros(w.size)

    # Flip vertically to convolve properly
    rev_w = np.flip(w)

    for i in range(w.size):
        curr_w = np.full(w.size, w[i])
        res = np.convolve(main_func(w), conv_func(rev_w - curr_w), mode='valid')
        results[i] = res

    return results


def add_noise(map):
    '''
    Adds poisson noise to a map
    :param map: the map to add noise to
    :return: the map with noise
    '''
    height = math.floor(map.size / map[0].size)
    width = map[0].size
    for i in range(height):
        for j in range(width):
            map[i][j] = np.random.poisson(map[i][j])
    return


def n(w, uP=0, temp=20.44):
    '''
    # Fermi-Dirac Function
    :param w: energy at point
    :param uP: electron chemical potential
    :param temp: temperature of experiment
    :return: Fermi-Dirac factor
    '''
    # Boltzmann's constant (meV/K)
    kB = 8.617333262 * 10 ** (-2)
    # h-bar: 6.582 * 10 ** (-13) (mev*s) # (Implicit bc already expressing energy)
    if w > 50: return 0 # Rounding approximation
    if w < -50: return 1
    return 1 / (math.exp((w - uP) / kB / temp) + 1)


n_vectorized = np.vectorize(n)



def secondary_electron_contribution_array(w_array, p, q, r, s):
    '''
    Model for secondary electron effect as sigmoid function
    :param w_array: energy array
    :param p: scale
    :param q: horizontal shift
    :param r: steepness
    :param s: vertical shift
    :return:
    '''

    return_array = np.zeros(w_array.size)

    # p is scale-up factor (0, inf), q is horizontal shift (-inf, inf), r is steepness (-inf, 0]
    for i in range(w_array.size):
        return_array[i] = p / (1 + math.exp(r * w_array[i] - r * q)) + s

    return return_array


def manualRedChi(data, fit, absSigmaSquared, DOF=1):
    '''
    Reduced-Chi Calculation
    :param data: true data
    :param fit: fitted predictions
    :param absSigmaSquared: variance (standard deviation squared)
    :param DOF: degrees of freedom (number points - number parameters)
    :return: reduced chi value (1 is good, >1 is bad, <1 is overfit)
    '''
    res = 0
    for i in range(data.size):
        res += (data[i] - fit[i]) ** 2 / absSigmaSquared[i]
    return res / DOF


def manualFTest(data, fit1, para1, fit2, para2, absSigmaSquared, n):
    '''
    F-Test Calculation for comparing nested models
    :param data: true data
    :param fit1: smaller model fitted predictions
    :param para1: number of parameters in smaller model
    :param fit2: larger model fitted predictions
    :param para2: number of parameters in larger model
    :param absSigmaSquared: variance (standard deviation squared)
    :param n: number of data points
    :return:
    '''
    # fit1 should be 'nested' within fit2
    if (para2 <= para1):
        return ValueError
    chi1 = manualRedChi(data, fit1, absSigmaSquared)
    chi2 = manualRedChi(data, fit2, absSigmaSquared)
    return ((chi1 - chi2) / (para2 - para1)) / (chi2 / (n - para2))


def gaussian_form_normalized(x, sigma, mu):
    '''
    Gaussian Function (Normalized)
    :param x: input
    :param sigma: Gaussian width
    :param mu: horizontal shift
    :return: Normalized Gaussian evaluated at input
    '''
    return 1 / (sigma * (2 * math.pi) ** 0.5) * math.e ** (-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_form(x, a, b, c):
    '''
    Gaussian Function (General)
    :param x: input
    :param a: scale
    :param b: horizontal shift
    :param c: width
    :return: Gaussian evaluated at input
    '''

    return a * math.e ** ((- (x - b) ** 2) / (2 * c ** 2))


def lorentz_form(x, a, b, c, d):
    '''
    Lorentz Function with vertical shift
    :param x: input
    :param a: scale
    :param b: horizontal shift
    :param c: width
    :param d: vertical shift
    :return: Lorentz with vertical shift evaluated at input
    '''
    return a * c / ((x - b) ** 2 + c ** 2) + d


def parabola(x, a, b, c):
    '''
    Parabola with no horizontal shift
    :param x:
    :param a:
    :param c:
    :return:
    '''
    return a * (x - b) ** 2 + c