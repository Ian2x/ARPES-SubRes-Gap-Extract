from EDC_funcs import *

##################################################
# EXTRACT GAP - TRAJECTORY
##################################################


super_state_trajectory = np.zeros(z_width)

# increase fitting speed by saving data
initial_ac_estimate_params, initial_ac_estimate_pcov = scipy.optimize.curve_fit(lorentz_form_w_secondary_electrons, w,
                                                                                inv_Z[0], bounds=(
        [0, -70, 0, 500, -70, 0, 0],
        [np.inf, 0, np.inf, 700, 0, 0.5, 100]))

last_a, last_b, last_c, last_p, last_q, last_r, last_s = initial_ac_estimate_params

super_state_trajectory[0] = last_b

for i in range(1, z_width):  # width
    fit_params_loop, pcov_loop = scipy.optimize.curve_fit(lorentz_form_w_secondary_electrons, w, inv_Z[i],
                                                          p0=(last_a, last_b, last_c, last_p, last_q, last_r, last_s),
                                                          maxfev=1000)
    last_a, last_b, last_c, last_p, last_q, last_r, last_s = fit_params_loop
    super_state_trajectory[i] = last_b


def trajectory_form(x, a, c, dk, k_error=0):
    return -((a * (x - k_error) ** 2 + c) ** 2 + dk ** 2) ** 0.5


initial_ac_estimate_params, initial_ac_estimate_pcov = scipy.optimize.curve_fit(trajectory_form, k,
                                                                                super_state_trajectory,
                                                                                bounds=(
                                                                                    [0, -np.inf, 0, -0.02],
                                                                                    [np.inf, 0, np.inf, 0.02]))

initial_a_estimate, initial_c_estimate, initial_dk_estimate, k_error = initial_ac_estimate_params
k = k - k_error
kf = (-initial_c_estimate / initial_a_estimate) ** 0.5

print("INITIAL AC PARAMS [a, c, dk, k shift:")
print(initial_ac_estimate_params)
print("\nINITIAL KF ESTIMATE:")
print(kf)
