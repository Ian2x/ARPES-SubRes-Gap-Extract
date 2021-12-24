from extract_ac_by_trajectory import *


##################################################
# EDC FITTING FUNCTIONS
##################################################

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
    peak_index = 0
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
    fit_start_index = peak_index
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
# Display Data
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
# Extract T and scale by fitting EDCs
##################################################

scale_trajectory = [1.99796644e+04, 2.07472464e+04, 2.15039820e+04, 2.09345046e+04,
 2.15485590e+04, 2.17238710e+04, 2.19742936e+04, 2.26583991e+04,
 2.37498260e+04, 2.28926322e+04, 2.26498127e+04, 2.21932616e+04,
 2.24252527e+04, 2.27899377e+04, 2.25025035e+04, 2.25194116e+04,
 2.21918824e+04, 2.28878406e+04, 2.30074542e+04, 2.35824246e+04,
 2.29085158e+04, 2.22252543e+04, 2.23005893e+04, 2.27259580e+04,
 2.27312528e+04, 2.27731669e+04, 2.22804632e+04, 2.23903946e+04,
 2.25316890e+04, 2.25619726e+04, 2.28857252e+04, 2.27154452e+04,
 2.23132323e+04, 2.22407890e+04, 2.23767103e+04, 2.26206648e+04,
 2.21507500e+04, 2.21934170e+04, 2.22449841e+04, 2.23406542e+04,
 2.25087789e+04, 2.23801791e+04, 2.18790544e+04, 2.16066130e+04,
 2.13467743e+04, 2.17807968e+04, 2.40752238e+04, 2.09150567e+04,
 2.02562210e+04, 2.00373737e+04, 2.02873344e+04, 2.00940931e+04,
 1.96615034e+04, 2.00158090e+04, 2.10918329e+04, 2.02515618e+04,
 1.95720740e+04, 1.97236053e+04, 1.98841729e+04, 1.98372198e+04,
 1.97957360e+04, 2.01912584e+04, 4.44246464e+04, 5.35844828e+04,
 2.13216890e+04, 1.94222577e+04, 1.86155623e+04, 1.85878045e+04,
 4.36428096e+04, 3.16363617e+04, 3.05626395e+04, 3.14919405e+04,
 5.27736970e+08, 5.27796840e+08, 5.28094019e+08, 9.66776632e+08,
 9.66776633e+08, 9.66776634e+08,9.66776635e+08, 9.66776635e+08,
 9.66776635e+08, 9.66776635e+08, 9.66776635e+08, 9.66776635e+08,
 9.66776636e+08, 9.66776636e+08, 9.66776636e+08, 9.66776636e+08,
 9.66776636e+08, 9.66776636e+08, 9.66776636e+08, 9.66776636e+08,
 9.66776636e+08, 9.66776636e+08, 9.66776636e+08, 9.66776636e+08,
 9.66776637e+08, 9.66776637e+08, 9.66776637e+08, 9.66776637e+08,
 9.66776637e+08, 9.66776637e+08, 9.66776637e+08, 9.66776637e+08,
 6.33070421e+08, 6.72380842e+08, 6.66110732e+08, 6.56360863e+08,
 6.41099222e+08, 9.97257292e+08, 9.97257292e+08, 9.97257292e+08]

T_trajectory = [1.08241410e-02, 7.06048841e-03, 7.04822483e-03, 8.74478719e-03,
 6.84978971e-03, 6.84449046e-03, 8.37417974e-03, 6.43702764e-03,
 4.99306117e-03, 1.14358462e-02, 8.76978985e-03, 5.15624144e-03,
 7.01203850e-03, 3.25414836e-03, 1.56811024e-02, 4.58466560e-03,
 7.87460494e-03, 3.47528317e-01, 3.37723081e-01, 5.13074036e-01,
 3.20246756e-01, 2.65375429e-02, 8.26857711e-03, 1.62484121e-01,
 2.18207497e-01, 2.84663049e-01, 7.76166922e-03, 1.25208455e-02,
 9.03268147e-03, 1.28401067e-02, 3.64574022e-03, 1.67853162e-02,
 6.97461654e-03, 1.11746750e-02, 7.39965577e-03, 1.74778776e-02,
 8.72405369e-03, 1.56121327e-02, 4.89803539e-03, 6.61314355e-03,
 1.15446338e-02, 1.42198233e-02, 3.56048220e-03, 1.13988194e-02,
 6.50765114e-03, 1.67958595e-01, 9.84034007e-01, 2.02896080e-01,
 1.15813626e-02, 6.07795145e-03, 4.59289314e-03, 2.83872694e-02,
 3.08495946e-02, 2.43543242e-01, 8.03474810e-01, 5.80018468e-01,
 3.95483750e-01, 4.32637558e-01, 5.74983078e-01, 6.69186865e-01,
 7.56150068e-01, 9.70676729e-01, 1.20416997e+00, 1.40668903e+00,
 9.92246515e-01, 4.89869560e-01, 1.99899806e-02, 6.77361754e-03,
 5.59255910e+00, 4.05654403e+00, 3.77246638e+00, 4.23192983e+00,
 7.50000000e+01, 7.50000000e+01, 7.50000000e+01, 7.50000000e+01,
 7.50000000e+01, 7.50000000e+01, 7.50000000e+01, 7.50000000e+01,
 7.50000000e+01, 7.50000000e+01, 7.50000000e+01, 7.50000000e+01,
 7.50000000e+01, 7.50000000e+01, 7.50000000e+01, 7.50000000e+01,
 7.50000000e+01, 7.50000000e+01, 7.50000000e+01, 7.50000000e+01,
 7.50000000e+01, 7.50000000e+01, 7.50000000e+01, 7.50000000e+01,
 7.50000000e+01, 7.50000000e+01, 7.50000000e+01, 7.50000000e+01,
 7.50000000e+01, 7.50000000e+01, 7.50000000e+01, 7.50000000e+01,
 1.54825487e-01, 1.82124630e-01, 1.59832812e-01, 1.58684161e-01,
 5.91796816e-02, 2.40668276e+00, 3.37937689e+00, 4.47823614e+00]
plt.plot(k, scale_trajectory)
plt.show()
plt.plot(k, T_trajectory)
plt.show()
'''
scale_trajectory = np.zeros(z_width)
T_trajectory = np.zeros(z_width)
scale_0, T_0, dk_0, s_0, a_0 = np.ones(5)
q_0, r_0, c_0 = np.zeros(3)
p_0 = 400
for i in range(z_width):
    temp_w, temp_slice, temp_fitting_sigma, temp_points_in_fit, temp_fit_start_index, temp_fit_end_index = EDC_prep(i)

    temp_params, temp_pcov = scipy.optimize.curve_fit(partial(spectrum_slice_array, fixed_k=k[i]), temp_w, temp_slice,
                                                      bounds=(
                                                          [1 / ONE_BILLION, 0, 0, 300, -35, 0, 0, 0, -np.inf],
                                                          [ONE_BILLION, 75, 75, 500, 0, 0.6, 100, np.inf, 0]),
                                                      p0=[scale_0, T_0, dk_0, p_0, q_0, r_0, s_0, a_0, c_0],
                                                      sigma=temp_fitting_sigma)
    scale_trajectory[i] = temp_params[0]
    T_trajectory[i] = temp_params[1]
    scale_0, T_0, dk_0, p_0, q_0, r_0, s_0, a_0, c_0 = temp_params
    print(i)
print(scale_trajectory)
print(T_trajectory)
quit()
'''

scale_trajectory = []
T_trajectory = []
scale_0 = 20000
T_0, dk_0, s_0, a_0 = np.ones(4)
q_0, r_0, c_0 = np.zeros(3)
p_0 = 400
temp_list = [12, 25, 37, 50, 62, 75, 87, 100]
for i in temp_list:
    temp_w, temp_slice, temp_fitting_sigma, temp_points_in_fit, temp_fit_start_index, temp_fit_end_index = EDC_prep(i)

    temp_params, temp_pcov = scipy.optimize.curve_fit(partial(spectrum_slice_array, fixed_k=k[i]), temp_w, temp_slice,
                                                      bounds=(
                                                          [10000, 0, 0, 300, -35, 0, 0, 0, -np.inf],
                                                          [40000, 75, 75, 500, 0, 0.6, 100, np.inf, 0]),
                                                      p0=[scale_0, T_0, dk_0, p_0, q_0, r_0, s_0, a_0, c_0],
                                                      sigma=temp_fitting_sigma)
    scale_trajectory.append(temp_params[0])
    T_trajectory.append(temp_params[1])
    scale_0, T_0, dk_0, p_0, q_0, r_0, s_0, a_0, c_0 = temp_params
    # plt.plot(temp_w, temp_slice)
    # plt.plot(temp_w, spectrum_slice_array(temp_w, *temp_params, k[i]))
    # plt.show()
    print(i)
print(scale_trajectory)
print(T_trajectory)
plt.plot(temp_list, scale_trajectory)
plt.show()
plt.plot(temp_list, T_trajectory)
plt.show()
quit()