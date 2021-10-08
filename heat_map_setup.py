from functools import *

from general_functions_and_constants import *
import matplotlib.pyplot as plt
import lmfit
import scipy.optimize
import scipy.integrate

##################################################
# DISPERSION SETUP
##################################################

# [Lattice Constant]
lattice_a = 4  # typically 3.6-4.2 Angstrom

# [Brillouin momentum]
brillouin_k = math.pi / lattice_a  # angstrom || # hbar * math.pi / lattice_a # g*m/s

# [Fermi momentum]
kf = 0.5092958179 * brillouin_k  # 0.5092958179
# Overwrite Fermi Momentum to exactly 0.4 (A^-1)
kf = 0.4

# Normal-state Dispersion Variables (ak^2+c)
c = -1000  # min(-50*dk, -50) # typically -1000 to -10,000 mev
a = -c / (kf ** 2)
# ------------------------------------------------
# [Energy pixel step size]
w_step = 1.5
# [Energy detectors array]
w = np.arange(-250, 125, w_step)


# w: (-45, 20), k: (-0.0125, 0.125)
# w: (-100, 40), k: (-0.03, 0.025)
# w: (-400, 0), k: (-0.05, 0.025)

# Convert w in meV to corresponding index
def w_as_index(input_w, w=w):
    return int(round((input_w - min(w)) / (max(w) - min(w)) * (w.size - 1)))


# [Angle uncertainty]: typically 0.1-0.2 degrees, using 0.045 degrees here
d_theta = 0.045 * math.pi / 180

# [Momentum pixel step size]
k_step = (1 / hbar) * math.sqrt(2 * mass_electron / speed_light / speed_light * (6176.5840329647)) * d_theta / (
        10 ** 10)

# [Momentum detectors array]
k = np.arange(kf - 0.04 * kf, kf + 0.04 * kf, k_step)
print("k_step: " + str(k_step) + " | mink: " + str(min(k)) + " | maxk: " + str(max(k)) + " | #steps: " + str(k.size))


# Convert k in A^-1 to corresponding index
def k_as_index(input_k, k=k):
    return int(round((input_k - min(k)) / (max(k) - min(k)) * (k.size - 1)))


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


# (http://ex7.iphy.ac.cn/downfile/32_PRB_57_R11093.pdf)
def A_BCS_2(k, w, dk=dk, T=T):
    return T / math.pi / ((w - e(k) - (dk ** 2) / (w + e(k))) ** 2 + T ** 2)


# Intensity Pre-factor
def Io(k):
    return 1;


# Full Composition Function (Knows a, c, dk, and T)
def Io_n_A_BCS(k, w):
    return Io(k) * n_vectorized(w) * A_BCS(k, w, a, c, dk, T)


def Io_n_A_BCS_2(k, w):
    return Io(k) * n_vectorized(w) * A_BCS_2(k, w)


# Intensity
def I(k, w):
    return energy_convolution_map(k, w, Io_n_A_BCS, R_vectorized, scaleup_factor)


# Normal-state Composition Function (dk=0, knows a, c, and T)
def norm_state_Io_n_A_BCS(k, w):
    return Io(k) * n_vectorized(w) * A_BCS(k, w, a, c, 0, T)


# Normal-state Intensity (dk=0, knows a, c, and T)
def norm_state_I(k, w):
    return energy_convolution_map(k, w, norm_state_Io_n_A_BCS, R_vectorized, scaleup_factor)

##################################################
# HEAT MAP
##################################################

# k and w values for plotting
X, Y = np.meshgrid(k, w)

# The Spectrum
Z = I(X, Y)
print("Z.size:", Z.shape, "\n")
add_noise(Z)

z_width = Z[0].size
z_height = int(Z.size / z_width)
kf_index = k_as_index(kf)
