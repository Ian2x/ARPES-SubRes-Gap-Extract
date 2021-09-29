from real_data import *

##################################################
# EXTRACT KF FROM MDC AT EF
##################################################
'''
fermi_mdc = np.zeros(z_width)
ef_index = w_as_index(0, w)

for i in range(z_width):
    fermi_mdc[i] = Z[z_height-ef_index-1][i]


# Dirac delta function
def dirac_delta(x, a, scale, hori_shift, vert_shift):
    return scale * math.e ** (- (((x - hori_shift) / a) ** 2)) + vert_shift


dirac_delta_vectorized = np.vectorize(dirac_delta)

kf_extract_params, kf_extract_cov = scipy.optimize.curve_fit(dirac_delta, k, fermi_mdc)
extracted_kf = kf_extract_params[2]
kf = extracted_kf

extracted_c = c
extracted_a = -extracted_c / (extracted_kf ** 2)

plt.plot(k, dirac_delta(k, *kf_extract_params), label='Fit')
plt.plot(k, fermi_mdc)
plt.legend()
plt.show()

print("extracted params: ", kf_extract_params)
print("extracted a: ", extracted_a)
'''

'''
# account for fermi distribution and energy convolution
def lorentz_convolved_trajectory_integrand(w_prime, fixed_w, a, b, c):
    return lorentz_form(w_prime, a, b, c) * R(math.fabs(w_prime-fixed_w), energy_conv_sigma)* n(w_prime)

def lorentz_convolved_trajectory_array(fixed_w, a, b, c):
    return_array = np.zeros(fixed_w.size)
    for i in range(fixed_w.size):
        return_array[i] = scipy.integrate.quad(lorentz_convolved_trajectory_integrand, fixed_w[i]-35, fixed_w[i]+35, args=(fixed_w[i], a, b, c))[0]
        # return_array[i] = scipy.integrate.quad(lorentz_convolved_trajectory_integrand, -np.inf, np.inf, args=(fixed_w[i], a, b, c))[0]
    return return_array

def trajectory_form(x, a, c, dk):
    return -((a * x ** 2 + c) ** 2 + dk ** 2) ** 0.5

trajectory_form_vectorized = np.vectorize(trajectory_form)


def extract_a_c_from_map(Z):
    width = Z[0].size
    height = int(Z.size / width)
    curr_k_index = 0

    trajectory = np.full(width, np.inf)

    last_a = 1
    last_b = 1
    last_c = 1

    avg_lorentz_scale = []

    while True:
        real_data_slice = np.zeros(height)
        for i in range(height):
            real_data_slice[i] = Z[height - 1 - i][curr_k_index]

        traj_peak_params, traj_peak_pcov = scipy.optimize.curve_fit(lorentz_convolved_trajectory_array, w, real_data_slice, p0=(last_a, last_b, last_c))

        if curr_k_index < 10:
            avg_lorentz_scale.append(traj_peak_params[0])
        elif traj_peak_params[0] < 0.95 * sum(avg_lorentz_scale) / len(avg_lorentz_scale):
            break


        trajectory[curr_k_index] = traj_peak_params[1]


        last_a = traj_peak_params[0]
        last_b = traj_peak_params[1]
        last_c = traj_peak_params[2]
        curr_k_index += 1
        print("=====================")
        print(curr_k_index)
        print(traj_peak_params)
        print(traj_peak_params[1])


    reduced_trajectory = []
    reduced_k = []
    for i in range(width):
        if trajectory[i]!=np.inf:
            reduced_trajectory.append(trajectory[i])
            reduced_k.append(k[i])



    a_c_params, a_c_pcov = scipy.optimize.curve_fit(trajectory_form, reduced_k, reduced_trajectory)

    im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])
    plt.colorbar(im)
    plt.plot(reduced_k, reduced_trajectory, label='trace')
    plt.legend()
    plt.show()

    plt.plot(reduced_k, trajectory_form_vectorized(reduced_k, *a_c_params))
    plt.plot(reduced_k, reduced_trajectory)
    plt.show()
    print(a_c_params)

    # change a, c to new values
    if (a_c_params[0] < 0):
        return (-a_c_params[0], -a_c_params[1])
    else:
        return (a_c_params[0], a_c_params[1])
    # how to know when to stop fitting trajectory --> first try fitting until peak is within 3 energy resolution of peak
    # see how accurately can extract a, c, and then how well gap can be obtained from there
    # compare to direct super state trajectory

a, c = extract_a_c_from_map(Z)
extracted_kf = math.sqrt(-c / a)
kf = extracted_kf
'''

a = 955.6632874098885
c = -46.169179840226974
kf = 0.21979794546060416

