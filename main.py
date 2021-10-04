from kf_from_norm_state import *

##################################################
# Z HEAT MAP INFO
##################################################

print("kf:", kf, "| index:", k_as_index(kf, k))
print("energy_conv_sigma:", energy_conv_sigma)
print("dk:", dk)
print("T:", T)


##################################################
# GAP EXTRACTION
##################################################

# NORMAN TECHNIQUE

# Requires knowledge of temperature and energy resolution --> should test resilience of these
# Also no k convolution

# Integrand of energy convolution integral
def energy_conv_integrand(integration_w, fixed_w, T, dk, a, c, fixed_k):
    return A_BCS(fixed_k, integration_w, a, c, dk, T) * R(math.fabs(integration_w - fixed_w), energy_conv_sigma) * n(
        integration_w)


# Secondary electron effect
def secondary_electron_contribution_array(w_array, p, q, r, s):
    return_array = np.zeros(w_array.size)

    # p is scale-up factor (0, inf), q is horizontal shift (-inf, inf), r is steepness (-inf, 0]
    for i in range(w_array.size):
        return_array[i] = p / (1 + math.exp(r * w_array[i] - r * q)) + s

    return return_array


# EDC slice function
def spectrum_slice_array(w_array, p, q, r, s, scale, T, dk, a, c, fixed_k):
    return_array = np.zeros(w_array.size)
    secondary_electrons_array = secondary_electron_contribution_array(w_array, p, q, r, s)
    for i in range(w_array.size):
        return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250,
                                                       args=(w_array[i], T, dk, a, c, fixed_k))[0] + \
                          secondary_electrons_array[i]
    return return_array


# [Left Shift Multiplier]: How far left to continue fit
left_shift_mult = 2

# [Fit Start Momentum]: Momentum to start fit (not indexed)
fit_start_k = 0
if a != 0:
    fit_start_k = math.sqrt((-left_shift_mult * dk - c) / a)
print('fit_start_k (not indexed): ', fit_start_k)

# Save previous fit guesses to accelerate fitting process (superconducting state)
last_p = 500
last_q = -15
last_r = 0.05
last_s = 0

last_scale = 1
last_T = 1
last_dk = 1

# For copy-pasting results
easy_print_array = []

# Plot spectrum (Might need to do origin lower sometimes)
im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])

plt.title("Reduced window")
plt.colorbar(im)
plt.xlabel('k ($A^{-1}$)')
plt.ylabel('w (mev)')

plt.show()

# /Users/ianhu/PycharmProjects/ARPES-SubRes-Gap-Extract
# np.savetxt("/Users/ianhu/Documents/ARPES CNN/Dataset 1 - c=-1000, sigma=15/"+str(round(dk,6))+".csv", Z, delimiter=",")

# Current index of EDC being fitted
curr_index = k_as_index(kf, k) + 5  # +1 is to account for -1 at start of loop

# Store gap size estimates from each index
gap_estimates = np.zeros(k_as_index(kf, k) + 1)

# Keep track of number of EDCs used
EDCs_used_count = 0
'''
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

    # Ignore noisy data
    fit_start_index = -1
    fit_end_index = -1

    for i in range(z_height):
        # Build EDC
        EDC[i] = Z[z_height - 1 - i][curr_index] # CHANGED z_height-1-i => i
        # Start fit at first index greater than min_fit_count
        if fit_start_index == -1:
            if EDC[i] >= min_fit_count:
                fit_start_index = i
        # End fit at at last index less than min_fit_count
        if EDC[i] >= min_fit_count:
            fit_end_index = i

    # Check that there is sufficient room for energy convolution
    min_indexes_from_edge = 3 * energy_conv_sigma / w_step
    if round(min_indexes_from_edge) > fit_start_index:
        print("WARNING: Insufficient room for energy conv on top; pushed fit start down")
        fit_start_index = round(min_indexes_from_edge)
    if round(z_height - 1 - min_indexes_from_edge) < fit_end_index:
        print("WARNING: Insufficient room for energy conv on bottom; pushed fit end up")
        fit_end_index = round(z_height - 1 - min_indexes_from_edge)

    # Points included in fit
    points_in_fit = fit_end_index - fit_start_index + 1  # include end point
    if points_in_fit < 5:
        print(
            "ERROR: Not enough points to do proper EDC fit. Suggestions: expand upper/lower energy bounds or increase gap size")
        print("Accepted points: ", points_in_fit)
        print("fit_start_index: ", fit_start_index)
        print("fit_end_index: ", fit_end_index)
        quit()

    # Create slice w/ low noise points
    low_noise_slice = np.zeros(points_in_fit)
    low_noise_w = np.zeros(points_in_fit)
    for i in range(points_in_fit):
        low_noise_slice[i] = Z[i + fit_start_index][curr_index]
        low_noise_w[i] = w[i + fit_start_index]
    low_noise_slice = list(reversed(low_noise_slice))
    print(low_noise_w)
    print(low_noise_slice)

    # Functions to fit ( Cheated kf || Cheated kf + No gap || No kf )
    fit_func_w_scale_T_dk = partial(spectrum_slice_array, a=a, c=c, fixed_k=curr_k)
    fit_func_w_scale_T = partial(spectrum_slice_array, dk=0, a=a, c=c, fixed_k=curr_k)

    # Scipy curve fit

    # Remove 0s from fitting sigma
    fitting_sigma = np.sqrt(low_noise_slice)
    for i in range(len(fitting_sigma)):
        if fitting_sigma[i] <= 0:
            fitting_sigma[i] = 1

    scipy_full_params, scipy_full_pcov = scipy.optimize.curve_fit(fit_func_w_scale_T_dk,
                                                                  low_noise_w, low_noise_slice,
                                                                  p0=[last_p, last_q, last_r, last_s, last_scale, last_T,
                                                                      last_dk], maxfev=2000,
                                                                  bounds=(
                                                                  [300, -40, 0., 0, 1 / ONE_BILLION, 0., 0.],
                                                                  [700, 10, 1, 200, ONE_BILLION, 75., 75.]),
                                                                  sigma=fitting_sigma)

      
    scipy_red_params, scipy_red_pcov = scipy.optimize.curve_fit(fit_func_w_scale_T, low_noise_w,
                                                                low_noise_slice, p0=[last_scale, last_T], maxfev=2000,
                                                                bounds=([scaleup_factor / 10, 0.],
                                                                        [scaleup_factor * 10, 50.]),
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

    last_p = scipy_full_params[0]
    last_q = scipy_full_params[1]
    last_r = scipy_full_params[2]
    last_s = scipy_full_params[3]

    last_scale = scipy_full_params[4]
    last_T = scipy_full_params[5]
    last_dk = scipy_full_params[6]

    # Print dk guesses
    print("dk:", scipy_full_params[6])
    print(scipy_full_params)

    # Save dk guess to array
    gap_estimates[curr_index] = last_dk

    # Compare gap size estimate to number of slices suggested
    curr_dk_guess = sum(gap_estimates) / (k_as_index(kf, k) - curr_index + 1)
    curr_k_start_suggestion = math.sqrt((-2 * curr_dk_guess - c) / a)

print("final k index: ", curr_index)
print(gap_estimates)
print(curr_dk_guess)
'''
##################################################
# EXPERIMENTAL SECTION
##################################################

pars = lmfit.Parameters()
pars.add('a', min=850, max=1050)  # a = 955.6632874098885
pars.add('c', min=-60, max=-30)  # c = -46.169179840226974
pars.add('p', min=300, max=700)
pars.add('q', min=-40, max=10)
pars.add('r', min=0, max=1)
pars.add('s', min=0, max=200)
pars.add('scale', min=1 / ONE_BILLION, max=ONE_BILLION)
pars.add('T', min=0, max=75)
pars.add('dk', min=0, max=75)

# Momentum to perform fit over
momentum_to_fit = [k_as_index(kf, k), k_as_index(kf, k) - 5, k_as_index(kf, k) - 10]
print(momentum_to_fit)

# EDC slices to be fitted
low_noise_slices = []
for i in range(len(momentum_to_fit)):
    # Build EDC
    EDC = np.zeros(z_height)
    for j in range(z_height):
        EDC[j] = Z[z_height - 1 - j][momentum_to_fit[i]]
    # Store EDC
    low_noise_slices.append(EDC)
print(low_noise_slices)

# Store EDC function for each momentum
EDC_func_array = []
for i in range(len(momentum_to_fit)):
    EDC_func_array.append(partial(spectrum_slice_array, fixed_k=k[momentum_to_fit[i]]))
print(EDC_func_array)

temp = partial(spectrum_slice_array, fixed_k=k[momentum_to_fit[i]])


def residual(p):
    '''
    residual_array = np.zeros(z_height)
    for i in range(len(momentum_to_fit)):
        temp_array = EDC_func_array[i](w, p['p'], p['q'], p['r'], p['s'], p['scale'], p['T'], p['dk'], p['a'], p['c']) - low_noise_slices[i]
        residual_array = np.add(residual_array, np.fabs(temp_array))
    print(len(residual_array))
    '''
    return temp(w, p['p'], p['q'], p['r'], p['s'], p['scale'], p['T'], p['dk'], p['a'], p['c']) - low_noise_slices[0]
    # return EDC_func_array[0](w, p['p'], p['q'], p['r'], p['s'], p['scale'], p['T'], p['dk'], p['a'], p['c']) - low_noise_slices[0]
    '''
    residual_array = []
    for i in range(len(momentum_to_fit)):
        residual_array = np.concatenate((residual_array, (
                    EDC_func_array[i](w, p['p'], p['q'], p['r'], p['s'], p['scale'], p['T'], p['dk'], p['a'], p['c']) -
                    low_noise_slices[i])))
    return residual_array
    '''


'''
        return fit_func_w_dk_scale_T(low_noise_selected_w, p['scale'], p['T'], p['dk']) - low_noise_slice

model1 = params['offset'] + x * params['slope1']
    model2 = params['offset'] + x * params['slope2']

    resid1 = dat1 - model1
    resid2 = dat2 - model2
    return np.concatenate((resid1, resid2))

'''

mini = lmfit.Minimizer(residual, pars, nan_policy='propagate', calc_covar=True)
# out1 = mini.minimize(method='nelder')
# kwargs = {"sigma": np.sqrt(low_noise_slice)}
# result = mini.minimize(method='leastsq', params=out1.params, args=kwargs)
print("ENTERING THE VOID...")
result = mini.minimize(method='leastsq')

print(lmfit.fit_report(result))

##################################################
# SHOW PLOT
##################################################
plt.tight_layout()
# plt.savefig('(11) Fitting EDCs.svg', format='svg')
plt.show()
print('\n\n')
for element in easy_print_array:
    print(element)
