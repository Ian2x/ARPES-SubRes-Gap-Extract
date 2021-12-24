from real_data import *

##################################################
# EXTRACT GAP - TRAJECTORY
##################################################

def lorentz_form_w_secondary_electrons(x, a, b, c, p, q, r, s):
    lorentz = lorentz_form(x, a, b, c, 0)
    secondary = secondary_electron_contribution_array(x, p, q, r, s)
    output = np.zeros(len(x))
    for i in range(len(x)):
        output[i] = lorentz[i] + secondary[i]
    return output


inv_Z = np.array([list(i) for i in zip(*Z)])
super_state_trajectory = np.zeros(z_width)
relative_scale_factors = np.zeros(z_width)

for i in range(0, z_width):
    relative_scale_factors[i] = sum(inv_Z[i])

relative_scale_factors_normalization_factor = sum(relative_scale_factors) / len(relative_scale_factors)

relative_scale_factors /= relative_scale_factors_normalization_factor
print(relative_scale_factors)
print(len(relative_scale_factors))

'''
# account for fermi distribution and energy convolution
def lorentz_convolved_trajectory_integrand(w_prime, fixed_w, a, b, c):
    return lorentz_form(w_prime, a, b, c) * R(math.fabs(w_prime-fixed_w), energy_conv_sigma)* n(w_prime)

def lorentz_convolved_trajectory_array(fixed_w, a, b, c):
    return_array = np.zeros(fixed_w.size)
    for i in range(fixed_w.size):
        return_array[i] = scipy.integrate.quad(lorentz_convolved_trajectory_integrand, fixed_w[i]-250, fixed_w[i]+250, args=(fixed_w[i], a, b, c))[0]
    return return_array
'''

# increase fitting speed by saving data
initial_ac_estimate_params, initial_ac_estimate_pcov = scipy.optimize.curve_fit(lorentz_form_w_secondary_electrons, w,
                                                                                inv_Z[0], bounds=(
    [0, -70, 0, 500, -70, 0, 0],
    [np.inf, 0, np.inf, 700, 0, 0.5, 100]))
last_a = initial_ac_estimate_params[0]
last_b = initial_ac_estimate_params[1]
last_c = initial_ac_estimate_params[2]
last_p = initial_ac_estimate_params[3]
last_q = initial_ac_estimate_params[4]
last_r = initial_ac_estimate_params[5]
last_s = initial_ac_estimate_params[6]

super_state_trajectory[0] = last_b

relative_scale_factors2= np.zeros(z_width)
relative_scale_factors2[0] = last_a

# end_index = None
# minimum_trace_stop_energy = -10  # mev
# minimum_trace_width_fraction = 0.5
for i in range(1, z_width):  # width
    fit_params_loop, pcov_loop = scipy.optimize.curve_fit(lorentz_form_w_secondary_electrons, w, inv_Z[i],
                                                          p0=(last_a, last_b, last_c, last_p, last_q, last_r, last_s),
                                                          maxfev=1000)
    # if not first round: if trace going back down OR if trace above fermi energy
    # if i != 1 and i > minimum_trace_width_fraction * z_width and (
    #         fit_params_loop[1] < last_b and fit_params_loop[1] + fit_params_loop[2] > minimum_trace_stop_energy) or \
    #         fit_params_loop[1] + fit_params_loop[2] > 0:
    #     end_index = i
    #     break
    last_a = fit_params_loop[0]
    last_b = fit_params_loop[1]
    last_c = fit_params_loop[2]
    last_p = fit_params_loop[3]
    last_q = fit_params_loop[4]
    last_r = fit_params_loop[5]
    last_s = fit_params_loop[6]
    super_state_trajectory[i] = last_b

    relative_scale_factors2[i] = last_a
    '''
    plt.plot(w, inv_Z[i])
    plt.plot(w, lorentz_form_w_secondary_electrons(w, *fit_params_loop))
    plt.show()
    '''
relative_scale_factors = relative_scale_factors2

# fractional_k = np.zeros(end_index)
# fractional_super_state_trajectory = np.zeros(end_index)
#
# for i in range(end_index):
#     fractional_k[i] = k[i]
#     fractional_super_state_trajectory[i] = super_state_trajectory[i]
#

def trajectory_form(x, a, c, dk, k_error=0):
    return -((a * (x - k_error) ** 2 + c) ** 2 + dk ** 2) ** 0.5


# initial_ac_estimate_params, initial_ac_estimate_pcov = scipy.optimize.curve_fit(trajectory_form, fractional_k,
#                                                                                 fractional_super_state_trajectory,
#                                                                                 bounds=(
#                                                                                 [0, -np.inf, 0, -0.02], [np.inf, 0, np.inf, 0.02]))


initial_ac_estimate_params, initial_ac_estimate_pcov = scipy.optimize.curve_fit(trajectory_form, k,
                                                                                super_state_trajectory,
                                                                                bounds=(
                                                                                [0, -np.inf, 0, -0.02], [np.inf, 0, np.inf, 0.02]))


initial_a_estimate = initial_ac_estimate_params[0]
initial_c_estimate = initial_ac_estimate_params[1]
initial_dk_estimate = initial_ac_estimate_params[2]
k_error = initial_ac_estimate_params[3]
k = k - k_error
# fractional_k = fractional_k - k_error
initial_kf_estimate = (-initial_c_estimate / initial_a_estimate) ** 0.5
kf = initial_kf_estimate

print("INITIAL AC PARAMS [a, c, dk, k shift:")
print(initial_ac_estimate_params)
print("\nINITIAL KF ESTIMATE:")
print(initial_kf_estimate)