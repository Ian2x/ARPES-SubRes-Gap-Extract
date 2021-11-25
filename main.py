from extract_gap_by_trajectory import *

#TODO: fit ignoring quasiparticle data, make scale independent for different momenta

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
# EDC FITTING FUNCTIONS
##################################################

# NORMAN TECHNIQUE

# Requires knowledge of temperature and energy resolution --> should test resilience of these
# Also no k convolution

# Integrand of energy convolution integral
def energy_conv_integrand(integration_w, fixed_w, T, dk, a, c, fixed_k):
    return A_BCS(fixed_k, integration_w, a, c, dk, T) * R(math.fabs(integration_w - fixed_w), energy_conv_sigma) * n(
        integration_w)


# EDC slice function
def spectrum_slice_array(w_array, scale, T, dk, p, q, r, s, a, c, fixed_k):
    return_array = np.zeros(w_array.size)
    for i in range(w_array.size):
        return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250,
                                                       args=(w_array[i], T, dk, a, c, fixed_k))[0]
    # add in secondary electrons
    secondary = secondary_electron_contribution_array(w_array, p, q, r, s)
    for i in range(w_array.size):
        return_array[i] = return_array[i] + secondary[i]
    return return_array


def EDC_prep(curr_index):
    # Energy Distribution Curve (slice data)
    EDC = np.zeros(z_height)

    # Ignore noisy data
    fit_start_index = -1
    fit_end_index = -1

    peak = 0
    peak_index=0
    min = np.inf
    min_index = 0

    for i in range(z_height):

        # Build EDC
        EDC[i] = Z[z_height - 1 - i][curr_index]

        # Start fit at first index greater than min_fit_count
        if fit_start_index == -1:
            if EDC[i] >= min_fit_count:
                fit_start_index = i
        # End fit at at last index less than min_fit_count
        if EDC[i] >= min_fit_count:
            fit_end_index = i

        if EDC[i] > peak:
            peak = EDC[i]
            peak_index = i

    for i in range(peak_index):
        if EDC[i] < min:
            min = EDC[i]
            min_index = i
    '''
    # UNNECESSARY FOR REAL DATA
    # Check that there is sufficient room for energy convolution 
    min_indexes_from_edge = 3 * energy_conv_sigma / w_step
    if round(min_indexes_from_edge) > fit_start_index:
        print("WARNING: Insufficient room for energy conv on top; pushed fit start down")
        fit_start_index = round(min_indexes_from_edge)
    if round(z_height - 1 - min_indexes_from_edge) < fit_end_index:
        print("WARNING: Insufficient room for energy conv on bottom; pushed fit end up")
        fit_end_index = round(z_height - 1 - min_indexes_from_edge)
    '''
    for i in range(peak_index, z_height):
        if EDC[i] > (peak + min) / 2:
            peak_index += 1

    one_side_w = np.zeros(peak_index)
    one_side_EDC = np.zeros(peak_index)

    for i in range(peak_index):
        one_side_w[i] = w[z_height - i - 1]
        one_side_EDC[i] = EDC[i]
    # print(one_side_w)
    # print(one_side_EDC)



    temp_fit_params, temp_fit_pcov = scipy.optimize.curve_fit(lorentz_form_shifted, one_side_w, one_side_EDC, bounds=([3 * (peak-min), -100, 0, 0.5 * min],[7 * (peak-min), 0, 100, 1.5 * min]))
    # plt.plot(one_side_w, one_side_EDC)
    # plt.plot(one_side_w, lorentz_form_shifted(one_side_w, *temp_fit_params))
    # plt.vlines(temp_fit_params[1] - 1.5 * temp_fit_params[2], 350, 450)
    # plt.show()
    fit_start_index = max(fit_start_index, w_as_index(temp_fit_params[1] - 1.5 * temp_fit_params[2] , w))

    # Points included in fit
    points_in_fit = fit_end_index - fit_start_index + 1  # include end point
    if points_in_fit < 5:
        print("Accepted points: ", points_in_fit)
        print("fit_start_index: ", fit_start_index)
        print("fit_end_index: ", fit_end_index)
        raise RuntimeError(
            "ERROR: Not enough points to do proper EDC fit. Suggestions: expand upper/lower energy bounds or increase gap size")

    # Create slice w/ low noise points
    low_noise_slice = np.zeros(points_in_fit)
    low_noise_w = np.zeros(points_in_fit)
    w_reversed = np.flip(w)
    for i in range(points_in_fit):
        low_noise_slice[i] = EDC[i + fit_start_index]
        low_noise_w[i] = w_reversed[i + fit_start_index]

    # Remove 0s from fitting sigma
    fitting_sigma = np.sqrt(low_noise_slice)
    for i in range(len(fitting_sigma)):
        if fitting_sigma[i] <= 0:
            fitting_sigma[i] = 1
    return (low_noise_w, low_noise_slice, fitting_sigma, points_in_fit, fit_start_index, fit_end_index)

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
fit_start_k = math.sqrt((-left_shift_mult * initial_dk_estimate - initial_c_estimate) / initial_a_estimate)
# fit_start_k = math.sqrt((-left_shift_mult * dk - c) / a)
print(fit_start_k)
quit()

fit_start_k = -initial_kf_estimate

print('fit_start_k (not indexed): ', -initial_kf_estimate)
print('fit_start_k (indexed): ', k_as_index(fit_start_k, k))

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
# pars.add('p', min=400, max=800)
# pars.add('q', min=-40, max=10)
# pars.add('r', min=-ONE_BILLION, max=ONE_BILLION)
pars.add('scale', value=scipy_full_params[0], min=1 / ONE_BILLION, max=ONE_BILLION)
pars.add('T', value=scipy_full_params[1], min=0, max=75)
pars.add('dk', value=scipy_full_params[2], min=0, max=75)
pars.add('p', value=initial_p_estimate, min=300, max=500)
pars.add('q', value=initial_q_estimate, min=-35, max=0)
pars.add('r', value=initial_r_estimate, min=0, max=0.6)
pars.add('s', value=initial_s_estimate, min=0, max=100)
pars.add('a', value=initial_a_estimate, min=0, max=np.inf)
pars.add('c', value=initial_c_estimate, min=-np.inf, max=0)

max_fit_start_index = k_as_index(kf, k) - 2
target_fit_start_index = k_as_index(fit_start_k, k)
if max_fit_start_index < target_fit_start_index:
    print("ERROR PRONE ZONE, LOW FITTING SPOTS USED")
    target_fit_start_index = max_fit_start_index

momentum_to_fit = [*range(target_fit_start_index, k_as_index(kf, k), 15)]

EDC_func_array = []
low_noise_slices = []
low_noise_ws = []
for i in range(len(momentum_to_fit)):
    temp_low_noise_w, temp_low_noise_slice, _, _, _, _ = EDC_prep(momentum_to_fit[i])
    low_noise_ws.append(temp_low_noise_w)
    low_noise_slices.append(temp_low_noise_slice)
    EDC_func_array.append(partial(spectrum_slice_array, fixed_k=k[momentum_to_fit[i]]))

print(momentum_to_fit)

def residual(p):
    residual = np.zeros(0)
    for i in range(len(momentum_to_fit)):
        EDC_residual = EDC_func_array[i](low_noise_ws[i], p['scale'], p['T'], p['dk'], p['p'], p['q'], p['r'], p['s'], p['a'], p['c']) - low_noise_slices[i]
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
    plt.title("momentum: " + k[momentum_to_fit[i]])
    plt.plot(low_noise_ws[i], low_noise_slices[i], label='data')
    plt.plot(low_noise_ws[i], EDC_func_array[i](low_noise_ws[i], lmfit_scale, lmfit_T, lmfit_dk, lmfit_p, lmfit_q, lmfit_r, lmfit_s, lmfit_a, lmfit_c), label='fit')
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