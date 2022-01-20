from real_data import *

##################################################
# EDC FITTING FUNCTIONS
##################################################

def energy_conv_integrand(integration_w, fixed_w, T, dk, a, c, temp, fixed_k):
    '''
    Integrand of energy convolution integral
    :param integration_w: w to integrate over
    :param fixed_w: w to calculate convolution about
    :param T: single-particle scattering rate
    :param dk: gap size
    :param a:
    :param c:
    :param fixed_k: momentum of EDC
    :return:
    '''
    return A_BCS(fixed_k, integration_w, a, c, dk, T) * R(math.fabs(integration_w - fixed_w), energy_conv_sigma) * n(
        integration_w, temp)


def spectrum_slice_array_SEC(w_array, scale, T, dk, p, q, r, s, a, c, temp, fixed_k):
    '''
    EDC slice function with secondary electron contribution
    :param w_array: energy array
    :param scale: scaling factor
    :param T: single-particle scattering rate
    :param dk: gap size
    :param p: SEC scale
    :param q: SEC horizontal shift
    :param r: SEC steepness
    :param s: SEC vertical shift
    :param a:
    :param c:
    :param fixed_k: momentum of EDC
    :return:
    '''
    return_array = np.zeros(w_array.size)
    for i in range(w_array.size):
        return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250,
                                                       args=(w_array[i], T, dk, a, c, temp, fixed_k))[0]
    # add in secondary electrons
    secondary = secondary_electron_contribution_array(w_array, p, q, r, s)
    for i in range(w_array.size):
        return_array[i] = return_array[i] + secondary[i]
    return return_array


def spectrum_slice_array(w_array, scale, T, dk, a, c, fixed_k):
    '''
    EDC slice function
    :param w_array: energy array
    :param scale: scaling factor
    :param T: single-particle scattering rate
    :param dk: gap size
    :param a:
    :param c:
    :param fixed_k: momentum of EDC
    :return:
    '''
    return_array = np.zeros(w_array.size)
    for i in range(w_array.size):
        return_array[i] = scale * scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250,
                                                       args=(w_array[i], T, dk, a, c, temp, fixed_k))[0]
    return return_array


def EDC_prep(curr_index, Z, w):
    '''
    Prepares relevant variables for EDC calculations
    :param curr_index: index of EDC
    :return: (low_noise_w, low_noise_slice, fitting_sigma, points_in_fit, fit_start_index, fit_end_index)
    '''
    z_height = len(Z)
    # Energy Distribution Curve (slice data)
    EDC = np.zeros(z_height)

    # Ignore noisy data
    fit_start_index = -1
    fit_end_index = -1

    peak = 0
    peak_index = 0
    min = np.inf

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

        if EDC[i] > peak:
            peak = EDC[i]
            peak_index = i

    for i in range(peak_index, z_height):
        if EDC[i] < min:
            min = EDC[i]
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
    '''
    one_side_w = np.zeros(peak_index)
    one_side_EDC = np.zeros(peak_index)

    for i in range(peak_index):
        one_side_w[i] = w[z_height - i - 1]
        one_side_EDC[i] = EDC[i]
    # print(one_side_w)
    # print(one_side_EDC)

    temp_fit_params, temp_fit_pcov = scipy.optimize.curve_fit(lorentz_form, one_side_w, one_side_EDC, bounds=(
    [3 * (peak - min), -100, 0, 0.5 * min], [7 * (peak - min), 0, 100, 1.5 * min]))
    # plt.plot(one_side_w, one_side_EDC)
    # plt.plot(one_side_w, lorentz_form_shifted(one_side_w, *temp_fit_params))
    # plt.vlines(temp_fit_params[1] - 1.5 * temp_fit_params[2], 350, 450)
    # plt.show()
    fit_start_index = max(fit_start_index, w_as_index(temp_fit_params[1] - 1.5 * temp_fit_params[2], w))
    '''
    fit_end_index = peak_index
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
        low_noise_w[i] = w[i + fit_start_index]

    # Remove 0s from fitting sigma
    fitting_sigma = np.sqrt(low_noise_slice)
    for i in range(len(fitting_sigma)):
        if fitting_sigma[i] <= 0:
            fitting_sigma[i] = 1
    return (low_noise_w, low_noise_slice, fitting_sigma, points_in_fit, fit_start_index, fit_end_index)

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