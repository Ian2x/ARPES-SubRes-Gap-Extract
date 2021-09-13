from general_functions_and_constants import *
import matplotlib.pyplot as plt
import lmfit
import scipy.optimize
import scipy.integrate

from functools import *

##################################################
# DISPERSION FUNCTIONS
##################################################
# normal-state dispersion
lattice_a = 4  # 2-5 angstrom (mainly 3.6-4.2)
brillouin_k = math.pi / lattice_a  # angstrom || # hbar * math.pi / lattice_a # g*m/s
fermi_k = 0.5092958179 * brillouin_k  # 0.5092958179
# overwrite fermi_k
fermi_k = 0.4

c = -1000  # min(-50*dk, -50) # -1000 to -10,000 mev
a = -c / (fermi_k ** 2)  # controlled by lattice_a and c

# print("c, a:",c, a)

# fermi momentum
kf = fermi_k  # math.fabs(c/a) ** 0.5

# ------------------------------------------------
w_step = 1.5
w = np.arange(-250, 125, w_step)


# w: (-45, 20), k: (-0.0125, 0.125)
# w: (-100, 40), k: (-0.03, 0.025)
# w: (-400, 0), k: (-0.05, 0.025)

def w_as_index(input_w):
    return int(round((input_w - min(w)) / (max(w) - min(w)) * (w.size - 1)))


d_theta = 0.045 * math.pi / 180  # 0.1-0.2 degrees  # from 0.045
k_step = (1 / hbar) * math.sqrt(2 * mass_electron / speed_light / speed_light * (6176.5840329647)) * d_theta / (
            10 ** 10)
k = np.arange(fermi_k - 0.04 * fermi_k, fermi_k + 0.04 * fermi_k, k_step)
print("k_step: " + str(k_step) + " | mink: " + str(min(k)) + " | maxk: " + str(max(k)) + " | #steps: " + str(k.size))


def k_as_index(input_k):
    return int(round((input_k - min(k)) / (max(k) - min(k)) * (k.size - 1)))


# ------------------------------------------------

def e(k, a, c):
    return a * k ** 2 + c


# band dispersion of BQPs
def E(k, a, c, dk):
    return (e(k, a, c) ** 2 + dk ** 2) ** 0.5


# coherence factors (relative intensity of BQP bands above and below EF)
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


# BCS spectral function
def A_BCS(k, w, a, c, dk, T):  # from (https://arxiv.org/pdf/cond-mat/0304505.pdf) (non-constant gap)
    return (1 / math.pi) * (
                u_vectorized(k, a, c, dk) * T / ((w - E(k, a, c, dk)) ** 2 + T ** 2) + v(k, a, c, dk) * T / (
                    (w + E(k, a, c, dk)) ** 2 + T ** 2))
    # return (1 / math.pi) * (u(k) * T / ((w - E(k, dk)) ** 2 + T ** 2) + v(k) * T / ((w + E(k, dk)) ** 2 + T ** 2))


def A_BCS_2(k, w, dk=dk, T=T):  # from (http://ex7.iphy.ac.cn/downfile/32_PRB_57_R11093.pdf)
    return T / math.pi / ((w - e(k) - (dk ** 2) / (w + e(k))) ** 2 + T ** 2)


# intensity pre-factor
def Io(k):
    return 1;


# composition function, using directly
def Io_n_A_BCS(k, w):
    return Io(k) * n_vectorized(w) * A_BCS(k, w, a, c, dk, T)


def Io_n_A_BCS_2(k, w):
    return Io(k) * n_vectorized(w) * A_BCS_2(k, w)


# intensity
def I(k, w):
    return energy_convolution_map(k, w, Io_n_A_BCS, R_vectorized, scaleup_factor)


def norm_state_Io_n_A_BCS(k, w):
    return Io(k) * n_vectorized(w) * A_BCS(k, w, a, c, 0, T)


def norm_state_I(k, w):
    return energy_convolution_map(k, w, norm_state_Io_n_A_BCS, R_vectorized, scaleup_factor)


# how far left to shift fit
left_shift_mult = 2
# fit_start_k = math.sqrt((-math.sqrt(left_shift_mult ** 2 - 1) * dk - c)/a)
# print(fit_start_k)
fit_start_k = 0
if a != 0:
    fit_start_k = math.sqrt((-left_shift_mult * dk - c) / a)
print('fit_start_k (not indexed): ', fit_start_k)

##################################################
# HEAT MAP
##################################################

X, Y = np.meshgrid(k, w)

Z = I(X, Y)
print("Z.size:", Z.shape, "\n")
add_noise(Z)

z_width = Z[0].size
z_height = int(Z.size / z_width)
kf_index = k_as_index(kf)

im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)],
                origin='lower')  # drawing the function
plt.colorbar(im)
plt.show()

##################################################
# EUGEN DATA GENERATION
##################################################


# counts as int



dk_selection = [
    1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2,
    8, 8, 8, 10, 10, 10, 12, 12, 12,
    36, 36, 36, 40, 40, 40, 44, 44, 44]
energy_conv_sigma_selection = [
    2.5, 3, 3.5, 2.5, 3, 3.5, 2.5, 3, 3.5,
    13, 15, 17, 13, 15, 17, 13, 15, 17,
    46, 50, 54, 46, 50, 54, 46, 50, 54]
T_selection = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
scaleup_factor_selection = [200, 400, 700, 1000, 2000, 4000, 7000, 10000, 15000, 20000]

progress_count = 0
for x in range(len(dk_selection)):
    for y in range(len(T_selection)):
        for z in range(len(scaleup_factor_selection)):

            progress_count+=1
            print(str(progress_count) + "/" + str(len(dk_selection)*len(T_selection)*len(scaleup_factor_selection)))

            dk = dk_selection[x]
            energy_conv_sigma = energy_conv_sigma_selection[x]
            T = T_selection[y]
            scaleup_factor = scaleup_factor_selection[z] * (energy_conv_sigma + T)


            Z = I(X, Y)
            add_noise(Z)
            Z = Z.astype(int)

            if x==0 and y==0:
                # plot 10 times
                # plt.cm.RdBu
                im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)],
                                origin='lower')  # drawing the function
                plt.colorbar(im)
                plt.show()

            file = open(r"/Users/ianhu/Documents/ARPES/Preliminary Coarse-Graining/" + "dk=" + str(dk) + ",ER=" + str(
                energy_conv_sigma) + ",T=" + str(T) + ",CF=" + str(round(scaleup_factor / (energy_conv_sigma + T), 2)) + ".txt", "w+")
            file.write("dk [meV]=" + str(dk) + '\n')
            file.write("energy res [meV]=" + str(energy_conv_sigma) + '\n')
            file.write("T [meV]=" + str(T) + '\n')
            file.write("Count factor [energy res+T]=" + str(scaleup_factor / (energy_conv_sigma + T)) + '\n\n')

            # print momentum [A^-1]
            file.write("Width name=Momentum [A^-1]\n")
            file.write("Width size=" + str(z_width) + '\n')
            for i in range(k.size):
                file.write(str(round(k[i], 3)) + '\t')
            file.write('\n\n')

            # print energy [meV]
            file.write("Height name=Energy [meV]\n")
            file.write("Height size=" + str(z_height) + '\n')
            for i in range(w.size):
                file.write(str(w[i]) + '\t')
            file.write('\n\n')

            for i in range(z_width):
                # print k
                file.write(str(round(k[i], 3)) + '\t')
                for j in range(z_height):
                    # print Z
                    file.write(str(Z[j][i]) + '\t')
                file.write('\n')

            file.close()
quit()
