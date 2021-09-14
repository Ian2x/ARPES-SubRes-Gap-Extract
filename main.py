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


# EDC slice function
def spectrum_slice_array(w_array, scale, T, dk, a, c, fixed_k):
    return_array = np.zeros(w_array.size)
    for i in range(w_array.size):
        return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250,
                                                       args=(w_array[i], T, dk, a, c, fixed_k))[0]
    return return_array


# Linearly space 6 points (max of 6 EDCs in fit)
short_k = np.linspace(max(0, k_as_index(fit_start_k)), k_as_index(extracted_kf, k), 6)
# Round to integers
short_k = short_k.astype(int)
# Remove repeats
short_k = np.unique(short_k)
# Try to fit from fermi momentum
short_k = np.flip(short_k, 0)
print(short_k)

# Save previous fit guesses to accelerate fitting process (superconducting state)
last_dk = 1
last_scale = scaleup_factor / 10
last_T = 1

# Save previous fit guesses to accelerate fitting process (normal state)
norm_state_last_scale = scaleup_factor / 10
norm_state_last_T = 1
norm_state_last_a = 1
norm_state_last_c = -1

# For copy-pasting results
easy_print_array = []

# Plot spectrum (Might need to do origin lower sometimes)
im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])

plt.colorbar(im)
plt.xlabel('k ($A^{-1}$)')
plt.ylabel('w (mev)')

plt.show()

print(Z)

# /Users/ianhu/PycharmProjects/ARPES-SubRes-Gap-Extract
# np.savetxt("/Users/ianhu/Documents/ARPES CNN/Dataset 1 - c=-1000, sigma=15/"+str(round(dk,6))+".csv", Z, delimiter=",")

# Current index of EDC being fitted
curr_index = k_as_index(kf, k) + 1  # +1 is to account for -1 at start of loop

# Store gap size estimates from each index
gap_estimates = np.zeros(k_as_index(kf, k) + 1)

# Keep track of number of EDCs used
EDCs_used = 0

# Run at least 3 times, then continue factoring in more EDC slices while suggested
while EDCs_used < 3 or curr_k_start_suggestion < k[curr_index]:
    EDCs_used += 1
    curr_index -= 1
    if (curr_index < 0):
        if (EDCs_used < 3):
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
        EDC[i] = Z[i][curr_index]
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

    # Functions to fit ( Cheated kf || Cheated kf + No gap || No kf )
    fit_func_w_scale_T_dk = partial(spectrum_slice_array, a=a, c=c, fixed_k=curr_k)
    fit_func_w_scale_T = partial(spectrum_slice_array, dk=0, a=a, c=c, fixed_k=curr_k)
    nokf_fit_func_w_scale_T_dk = partial(spectrum_slice_array, a=extracted_a, c=extracted_c, fixed_k=curr_k)

    # Scipy curve fit

    # Remove 0s from fitting sigma
    fitting_sigma = np.sqrt(low_noise_slice)
    for i in range(len(fitting_sigma)):
        if fitting_sigma[i] <= 0:
            fitting_sigma[i] = 1

    scipy_full_params, scipy_full_pcov = scipy.optimize.curve_fit(fit_func_w_scale_T_dk,
                                                                  low_noise_w, low_noise_slice,
                                                                  p0=[last_scale, last_T, last_dk], maxfev=2000,
                                                                  bounds=([scaleup_factor / 10, 0., 0.],
                                                                          [scaleup_factor * 10, 50., 50.]),
                                                                  sigma=fitting_sigma)
    scipy_red_params, scipy_red_pcov = scipy.optimize.curve_fit(fit_func_w_scale_T, low_noise_w,
                                                                low_noise_slice, p0=[last_scale, last_T], maxfev=2000,
                                                                bounds=([scaleup_factor / 10, 0.],
                                                                        [scaleup_factor * 10, 50.]),
                                                                sigma=fitting_sigma)
    nokf_scipy_full_params, nokf_scipy_full_pcov = scipy.optimize.curve_fit(nokf_fit_func_w_scale_T_dk,
                                                                            low_noise_w, low_noise_slice,
                                                                            p0=[last_scale, last_T, last_dk],
                                                                            maxfev=2000,
                                                                            bounds=([scaleup_factor / 10, 0., 0.],
                                                                                    [scaleup_factor * 10, 50., 50.]),
                                                                            sigma=fitting_sigma)

    # Plot fits
    # plt.plot(w, fit_func_w_scale_T_dk(w, *scipy_full_params), label='Fitted curve (cheated)')
    plt.plot(w, nokf_fit_func_w_scale_T_dk(w, *nokf_scipy_full_params), label='Fitted curve (no kf)')
    plt.plot(w, fit_func_w_scale_T_dk(w, scaleup_factor / w_step, T, dk), label='Perfect fit')

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
    last_scale = nokf_scipy_full_params[0]
    last_T = nokf_scipy_full_params[1]
    last_dk = nokf_scipy_full_params[2]

    # Print dk guesses
    print("calculated dk (nokf):", last_dk)
    print("cheated dk (kf):", scipy_full_params[2])

    # Save dk guess to array
    gap_estimates[curr_index] = last_dk

    # Compare gap size estimate to number of slices suggested
    curr_dk_guess = sum(gap_estimates) / (k_as_index(kf, k) - curr_index + 1)
    curr_k_start_suggestion = math.sqrt((-2 * curr_dk_guess - extracted_c) / extracted_a)

print("final k index: ", curr_index)
print(gap_estimates)
print(curr_dk_guess)

##################################################
# SHOW PLOT
##################################################
plt.tight_layout()
# plt.savefig('(11) Fitting EDCs.svg', format='svg')
plt.show()
print('\n\n')
for element in easy_print_array:
    print(element)
