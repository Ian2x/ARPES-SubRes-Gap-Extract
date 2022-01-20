from extract_Tscale_by_EDCs import *

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

# plt.title("Raw Eugen data (Reduced Window)")
# im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])  # drawing the function
# plt.colorbar(im)
# # plt.plot(fractional_k, fractional_super_state_trajectory)
# plt.plot(k, trajectory_form(k, initial_a_estimate, initial_c_estimate, initial_dk_estimate))
# plt.show()


##################################################
# GET INITIAL GAP SIZE ESTIMATE --> WIDTH OF 2D FIT
##################################################

'''
kf_low_noise_w, kf_low_noise_slice, kf_fitting_sigma, kf_points_in_fit, kf_fit_start_index, kf_fit_end_index = EDC_prep(k_as_index(kf, k), Z, w)

kf_EDC_fit_formula = partial(spectrum_slice_array_SEC, a = initial_a_estimate, c = initial_c_estimate, fixed_k=k[k_as_index(kf, k=k)])
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
'''

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


pars = lmfit.Parameters()
'''
[[Variables]]
    a:  2.3649e+09 +/- 5.1427e+08 (21.75%) (init = 0)
    b:  1.5255e+08 +/- 22399302.7 (14.68%) (init = 0)
    c: -26737794.8 +/- 3738258.86 (13.98%) (init = 0)
    d: -2567192.51 +/- 124167.899 (4.84%) (init = 0)
    e: -29517.7140 +/- 6151.75967 (20.84%) (init = 0)
    f:  63331.3795 +/- 124.183427 (0.20%) (init = 0)
[-2.43734497e+07  6.13594963e+05  2.79541030e+05 -8.07612271e+03
 -1.50010981e+03  2.71522060e+01  1.11920804e+01]
'''
pars.add('scale_a', value=2.3649e+09, min=7.2e+08, max=7.2e+09)
pars.add('scale_b', value=1.5255e+08, min=5.0e+07, max=5.0e+08)
pars.add('scale_c', value=-2.67377948e+07, min=-8.6e+07, max=-8.6e+06)
pars.add('scale_d', value=-2.56719251e+06, min=-8.6e+06, max=-8.6e+05)
pars.add('scale_e', value=-2.95177140e+04, min=-1.0e+05, max=-1.0e+04)
pars.add('scale_f', value=6.33313795e+04, min=1.9e+04, max=1.9e+05)
pars.add('T_a', value=-2.43734497e+07, min=-7.9e+07, max=-7.9e+06)
pars.add('T_b', value=6.13594963e+05, min=2.0e+05, max=2.0e+06)
pars.add('T_c', value=2.79541030e+05, min=9.2e+04, max=9.2e+05)
pars.add('T_d', value=-8.07612271e+03, min=-2.7e+04, max=-2.7e+03)
pars.add('T_e', value=-1.50010981e+03, min=-5.0e+03, max=-5.0e+02)
pars.add('T_f', value=2.71522060e+01, min=0, max=50)
pars.add('T_g', value=1.11920804e+01, min=0, max=50)
pars.add('temp', value=20.44, vary=False)
pars.add('dk', value=15, min=0, max=75)
pars.add('p', value=48, min=0, max=480)
pars.add('q', value=-18, min=-180, max=0)
pars.add('r', value=0.38, min=0, max=3.8)
pars.add('s', value=-14, min=-140, max=0)
pars.add('a', value=initial_a_estimate, min=0, max=initial_a_estimate*3)
pars.add('c', value=initial_c_estimate, min=initial_c_estimate*3, max=0)

# max_fit_start_index = k_as_index(kf, k) - 2
# target_fit_start_index = k_as_index(fit_start_k, k)
# if max_fit_start_index < target_fit_start_index:
#     print("ERROR PRONE ZONE, LOW FITTING SPOTS USED")
#     target_fit_start_index = max_fit_start_index

momentum_to_fit = [*range(k_as_index(-kf, k), k_as_index(-fit_start_k, k), 30)] + [*range(k_as_index(fit_start_k, k), k_as_index(kf, k), 30)]
momentum_to_fit = [5, 10, 95, 100]
print(momentum_to_fit)

EDC_func_array = []
low_noise_slices = []
low_noise_ws = []
for i in range(len(momentum_to_fit)):
    ki = momentum_to_fit[i]
    temp_low_noise_w, temp_low_noise_slice, _, _, _, _ = EDC_prep(ki, Z, w)
    low_noise_ws.append(temp_low_noise_w)
    low_noise_slices.append(temp_low_noise_slice)
    EDC_func_array.append(partial(spectrum_slice_array_SEC, fixed_k=math.fabs(k[ki])))
    # EDC_func_array.append(partial(spectrum_slice_array, fixed_k=math.fabs(k[ki])))




'''
short_relative_scale_factors = np.zeros(len(momentum_to_fit))
for i in range(len(momentum_to_fit)):
    short_relative_scale_factors[i] = relative_scale_factors[momentum_to_fit[i]]
short_relative_scale_factors_normalization_factor = sum(short_relative_scale_factors) / len(short_relative_scale_factors)
short_relative_scale_factors /= short_relative_scale_factors_normalization_factor
'''

def residual(p):
    residual = np.zeros(0)
    for i in range(len(momentum_to_fit)):
        ki = momentum_to_fit[i]
        local_scale = d6_polynomial(k[ki], p['scale_a'], p['scale_b'], p['scale_c'], p['scale_d'], p['scale_e'], p['scale_f'])
        local_T = d7_polynomial(k[ki], p['T_a'], p['T_b'], p['T_c'], p['T_d'], p['T_e'], p['T_f'], p['T_g'])
        EDC_residual = EDC_func_array[i](low_noise_ws[i], local_scale, local_T, p['dk'], p['p'], p['q'], p['r'], p['s'], p['a'], p['c'], p['temp']) - low_noise_slices[i]
        # EDC_residual = EDC_func_array[i](low_noise_ws[i], local_scale, local_T, p['dk'], p['a'], p['c']) - low_noise_slices[i]
        weighted_EDC_residual = EDC_residual / np.sqrt(low_noise_slices[i])
        residual = np.concatenate((residual, weighted_EDC_residual))
    return residual

mini = lmfit.Minimizer(residual, pars, nan_policy='propagate', calc_covar=True)
# out1 = mini.minimize(method='nelder')
# kwargs = {"sigma": np.sqrt(low_noise_slice)}
# result = mini.minimize(method='leastsq', params=out1.params, args=kwargs)
result = mini.minimize(method='leastsq')
print(lmfit.fit_report(result))
lmfit_scale_a = result.params.get('scale_a').value
lmfit_scale_b = result.params.get('scale_b').value
lmfit_scale_c = result.params.get('scale_c').value
lmfit_scale_d = result.params.get('scale_d').value
lmfit_scale_e = result.params.get('scale_e').value
lmfit_scale_f = result.params.get('scale_f').value
lmfit_T_a = result.params.get('T_a').value
lmfit_T_b = result.params.get('T_b').value
lmfit_T_c = result.params.get('T_c').value
lmfit_T_d = result.params.get('T_d').value
lmfit_T_e = result.params.get('T_e').value
lmfit_T_f = result.params.get('T_f').value
lmfit_T_g = result.params.get('T_g').value
lmfit_temp = result.params.get('temp').value
lmfit_dk = result.params.get('dk').value
lmfit_p = result.params.get('p').value
lmfit_q = result.params.get('q').value
lmfit_r = result.params.get('r').value
lmfit_s = result.params.get('s').value
lmfit_a = result.params.get('a').value
lmfit_c = result.params.get('c').value

print("FINAL DK: ")
# 15.084723755346952
print(lmfit_dk)


for i in range(len(momentum_to_fit)):
    ki = momentum_to_fit[i]
    plt.title("momentum: " + str(k[ki]))
    plt.plot(low_noise_ws[i], low_noise_slices[i], label='data')
    local_scale = d6_polynomial(k[ki], lmfit_scale_a, lmfit_scale_b, lmfit_scale_c, lmfit_scale_d, lmfit_scale_e, lmfit_scale_f)
    local_T = d7_polynomial(k[ki], lmfit_T_a, lmfit_T_b, lmfit_T_c, lmfit_T_d, lmfit_T_e, lmfit_T_f, lmfit_T_g)
    # plt.plot(low_noise_ws[i], EDC_func_array[i](low_noise_ws[i], local_scale, local_T, lmfit_dk, lmfit_a, lmfit_c), label='fit')
    plt.plot(low_noise_ws[i], EDC_func_array[i](low_noise_ws[i], local_scale, local_T, lmfit_dk, lmfit_p, lmfit_q, lmfit_r, lmfit_s, lmfit_a, lmfit_c, lmfit_temp), label='fit')
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