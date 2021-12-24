from extract_Tscale_by_EDCs import *

#TODO: 1) get a and c estimate with trajectory fit

# fit from -kf to -stop, stop to kf; unless stop such that should do -kf to kf
# rescale to make scale approx. same
# fit lorentz? somehow figure out quasiparticle edge
##################################################
# Z HEAT MAP INFO
##################################################

print("\nkf:", kf, "| index:", k_as_index(kf, k))
print("energy_conv_sigma:", energy_conv_sigma)
# print("dk:", dk)
# print("T:", T)

plt.title("Raw Eugen data (Reduced Window)")
im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])  # drawing the function
plt.colorbar(im)
# plt.plot(fractional_k, fractional_super_state_trajectory)
plt.plot(k, trajectory_form(k, initial_a_estimate, initial_c_estimate, initial_dk_estimate))
plt.show()


##################################################
# GET INITIAL GAP SIZE ESTIMATE --> WIDTH OF 2D FIT
##################################################


kf_low_noise_w, kf_low_noise_slice, kf_fitting_sigma, kf_points_in_fit, kf_fit_start_index, kf_fit_end_index = EDC_prep(
    k_as_index(kf, k))

kf_EDC_fit_formula = partial(spectrum_slice_array, a = initial_a_estimate, c = initial_c_estimate, fixed_k=k[k_as_index(kf, k=k)])
# def spectrum_slice_array(w_array, scale, T, dk, p, q, r, s, a, c, fixed_k):
# p is scale-up factor (0, inf), q is horizontal shift (-inf, inf), r is steepness (-inf, 0]
scipy_full_params, scipy_full_pcov = scipy.optimize.curve_fit(kf_EDC_fit_formula,
                                                              kf_low_noise_w, kf_low_noise_slice, maxfev=2000,
                                                              bounds=(
                                                                  [1 / ONE_BILLION, 0, 0, 300, -35, 0, 0],
                                                                  [ONE_BILLION, 75, 75, 500, 0, 0.6, 100]),
                                                              sigma=kf_fitting_sigma)
initial_scale_estimate, initial_T_estimate, initial_dk_estimate, initial_p_estimate, initial_q_estimate, initial_r_estimate, initial_s_estimate = scipy_full_params
plt.plot(kf_low_noise_w, kf_low_noise_slice, label='data')
plt.plot(kf_low_noise_w, kf_EDC_fit_formula(kf_low_noise_w, *scipy_full_params))
plt.title("kf initial fit")
plt.legend()
plt.show()
print("SCIPY PARAMS [scale, T, dk, p, q, r, s]:")
print(scipy_full_params)

initial_scale_estimate = scipy_full_params[0]
initial_T_estimate = scipy_full_params[1]
initial_dk_estimate = scipy_full_params[2]


# [Left Shift Multiplier]: How far left to continue fit
left_shift_mult = 2

# [Fit Start Momentum]: Momentum to start fit (not indexed)
try:
    fit_start_k = math.sqrt((-left_shift_mult * initial_dk_estimate - initial_c_estimate) / initial_a_estimate)
except ValueError:
    fit_start_k = 0
    print("Able to fit momenta through k=0")
# fit_start_k = math.sqrt((-left_shift_mult * dk - c) / a)
# quit()

# fit_start_k = -initial_kf_estimate

# print('fit_start_k (not indexed): ', -initial_kf_estimate)
# print('fit_start_k (indexed): ', k_as_index(fit_start_k, k))

# Plot spectrum
'''
im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])

plt.title("Reduced window")
plt.colorbar(im)
plt.xlabel('k ($A^{-1}$)')
plt.ylabel('w (mev)')
plt.show()
'''

##################################################
# 2D Fit (LMFIT)
##################################################

# def spectrum_slice_array(w_array, scale, T, dk, a, c, fixed_k):
#     return_array = np.zeros(w_array.size)
#     for i in range(w_array.size):
#         return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250,
#                                                        args=(w_array[i], T, dk, a, c, fixed_k))[0]
#     return return_array

pars = lmfit.Parameters()
pars.add('scale', value=scipy_full_params[0], min=1 / ONE_BILLION, max=ONE_BILLION)
pars.add('T', value=scipy_full_params[1], min=0, max=75)
pars.add('dk', value=scipy_full_params[2], min=0, max=75)
pars.add('p', value=initial_p_estimate, min=300, max=500)
pars.add('q', value=initial_q_estimate, min=-35, max=0)
pars.add('r', value=initial_r_estimate, min=0, max=0.6)
pars.add('s', value=initial_s_estimate, min=0, max=100)
pars.add('a', value=initial_a_estimate, min=0, max=np.inf)
pars.add('c', value=initial_c_estimate, min=-np.inf, max=0)

# max_fit_start_index = k_as_index(kf, k) - 2
# target_fit_start_index = k_as_index(fit_start_k, k)
# if max_fit_start_index < target_fit_start_index:
#     print("ERROR PRONE ZONE, LOW FITTING SPOTS USED")
#     target_fit_start_index = max_fit_start_index

momentum_to_fit = [*range(k_as_index(-initial_kf_estimate, k), k_as_index(-fit_start_k, k), 30)] + [*range(k_as_index(fit_start_k, k), k_as_index(initial_kf_estimate, k), 30)]
print(momentum_to_fit)

EDC_func_array = []
low_noise_slices = []
low_noise_ws = []
for i in range(len(momentum_to_fit)):
    temp_low_noise_w, temp_low_noise_slice, _, _, _, _ = EDC_prep(momentum_to_fit[i])
    low_noise_ws.append(temp_low_noise_w)
    low_noise_slices.append(temp_low_noise_slice)
    EDC_func_array.append(partial(spectrum_slice_array, fixed_k=k[momentum_to_fit[i]]))

short_relative_scale_factors = np.zeros(len(momentum_to_fit))
for i in range(len(momentum_to_fit)):
    short_relative_scale_factors[i] = relative_scale_factors[momentum_to_fit[i]]
short_relative_scale_factors_normalization_factor = sum(short_relative_scale_factors) / len(short_relative_scale_factors)
short_relative_scale_factors /= short_relative_scale_factors_normalization_factor



def residual(p):
    residual = np.zeros(0)
    for i in range(len(momentum_to_fit)):
        EDC_residual = short_relative_scale_factors[i]*EDC_func_array[i](low_noise_ws[i], p['scale'], p['T'], p['dk'], p['p'], p['q'], p['r'], p['s'], p['a'], p['c']) - low_noise_slices[i]
        # EDC_residual = short_relative_scale_factors[i]*EDC_func_array[i](low_noise_ws[i], p['scale'], p['T'], p['dk'], p['a'], p['c']) - low_noise_slices[i]
        residual = np.concatenate((residual, EDC_residual))
    return residual

mini = lmfit.Minimizer(residual, pars, nan_policy='propagate', calc_covar=True)
# out1 = mini.minimize(method='nelder')
# kwargs = {"sigma": np.sqrt(low_noise_slice)}
# result = mini.minimize(method='leastsq', params=out1.params, args=kwargs)
result = mini.minimize(method='leastsq')
print(lmfit.fit_report(result))
lmfit_scale = result.params.get('scale').value
lmfit_T = result.params.get('T').value
lmfit_dk = result.params.get('dk').value
lmfit_p = result.params.get('p').value
lmfit_q = result.params.get('q').value
lmfit_r = result.params.get('r').value
lmfit_s = result.params.get('s').value
lmfit_a = result.params.get('a').value
lmfit_c = result.params.get('c').value





for i in range(len(momentum_to_fit)):
    plt.title("momentum: " + str(k[momentum_to_fit[i]]))
    plt.plot(low_noise_ws[i], low_noise_slices[i], label='data')
    plt.plot(low_noise_ws[i], short_relative_scale_factors[i]*EDC_func_array[i](low_noise_ws[i], lmfit_scale, lmfit_T, lmfit_dk, lmfit_p, lmfit_q, lmfit_r, lmfit_s, lmfit_a, lmfit_c), label='fit')
    # plt.plot(low_noise_ws[i], short_relative_scale_factors[i]*EDC_func_array[i](low_noise_ws[i], lmfit_scale, lmfit_T, lmfit_dk, lmfit_a, lmfit_c), label='fit')
    plt.show()

'''
# Current index of EDC being fitted
curr_index = k_as_index(kf, k) + 5  # +1 is to account for -1 at start of loop

# Store gap size estimates from each index
gap_estimates = np.zeros(k_as_index(kf, k) + 1)

# Keep track of number of EDCs used
EDCs_used_count = 0

# Run at least 3 times, then continue factoring in more EDC slices while suggested
while EDCs_used_count < 3 or curr_k_start_suggestion < k[curr_index]:
    EDCs_used_count += 1
    curr_index -= 5
    if (curr_index < 0):
        if (EDCs_used_count < 3):
            print("ERROR: Not enough points left of kf to fit with")
            quit()
        print("WARNING: Hit momentum index 0 in fit, consider extending left bounds")
        break

    # Energy Distribution Curve (slice data)
    EDC = np.zeros(z_height)
    # Translate momentum index to momentum value
    curr_k = k[curr_index]

    print('==============================================')
    print('slice_k_index: ' + str(curr_index) + ' (' + str(curr_k) + ')')

    low_noise_w, low_noise_slice, fitting_sigma, points_in_fit, fit_start_index, fit_end_index = EDC_prep(curr_index)

    # Functions to fit ( Cheated kf || Cheated kf + No gap || No kf )
    fit_func_w_scale_T_dk = partial(spectrum_slice_array, a=a, c=c, fixed_k=curr_k)
    fit_func_w_scale_T = partial(spectrum_slice_array, dk=0, a=a, c=c, fixed_k=curr_k)

    # Scipy curve fit

    scipy_full_params, scipy_full_pcov = scipy.optimize.curve_fit(fit_func_w_scale_T_dk,
                                                                  low_noise_w, low_noise_slice, maxfev=2000,
                                                                  bounds=(
                                                                      [1 / ONE_BILLION, 0., 0.],
                                                                      [ONE_BILLION, 75., 75.]),
                                                                  sigma=fitting_sigma)

    # Plot fits
    plt.plot(w, fit_func_w_scale_T_dk(w, *scipy_full_params), label='Fitted curve')
    # plt.plot(w, fit_func_w_scale_T_dk(w, scaleup_factor / w_step, T, dk), label='Perfect fit')

    # Plot data
    plt.plot(w, EDC, label='Data')

    # Plot low_noise criteria
    plt.vlines(w[fit_start_index], 0, min_fit_count, color='black')
    plt.vlines(w[fit_end_index], 0, min_fit_count, color='black')

    # Plot title and legend
    # plt.title("k ($A^{-1}$): " + str(round(curr_k, 3)) + " | dk estimate:" + str(round(last_dk, 2)))
    plt.xlabel('w (mev)')
    plt.ylabel('counts')
    plt.legend()

    plt.show()

    # Save fit params to accelerate future fitting

    # Print dk guesses
    print("dk:", scipy_full_params[1])
    print(scipy_full_params)

    # Save dk guess to array
    gap_estimates[curr_index] = scipy_full_params[1]

    # Compare gap size estimate to number of slices suggested
    curr_dk_guess = sum(gap_estimates) / (k_as_index(kf, k) - curr_index + 1)
    curr_k_start_suggestion = math.sqrt((-2 * curr_dk_guess - c) / a)

print("final k index: ", curr_index)
print(gap_estimates)
print(curr_dk_guess)
'''