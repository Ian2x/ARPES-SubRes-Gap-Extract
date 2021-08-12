##################################################
# EXTRACT GAP - TRAJECTORY
##################################################
'''
inv_Z = np.array([list(i) for i in zip(*Z)])
super_state_trajectory = np.zeros(z_width)

# account for fermi distribution and energy convolution
def lorentz_convolved_trajectory_integrand(w_prime, fixed_w, a, b, c):
    return lorentz_form(w_prime, a, b, c) * R(math.fabs(w_prime-fixed_w), energy_conv_sigma)* n(w_prime)

def lorentz_convolved_trajectory_array(fixed_w, a, b, c):
    return_array = np.zeros(fixed_w.size)
    for i in range(fixed_w.size):
        return_array[i] = scipy.integrate.quad(lorentz_convolved_trajectory_integrand, fixed_w[i]-35, fixed_w[i]+35, args=(fixed_w[i], a, b, c))[0]
    return return_array


# increase fitting speed by saving data
fit_params_7, pcov_7 = scipy.optimize.curve_fit(lorentz_convolved_trajectory_array, w, remove_n(w, inv_Z[0]))
last_a=fit_params_7[0]
last_b=fit_params_7[1]
last_c=fit_params_7[2]
super_state_trajectory[0] = last_b

for i in range(z_width-1): # width
    i+=1
    vertical_slice = remove_n(w, inv_Z[i])
    fit_params_loop, pcov_loop = scipy.optimize.curve_fit(lorentz_convolved_trajectory_array, w, vertical_slice, p0=(last_a, last_b, last_c), maxfev=1000)
    last_a=fit_params_loop[0]
    last_b=fit_params_loop[1]
    last_c=fit_params_loop[2]
    super_state_trajectory[i] = last_b

plt.plot(k, super_state_trajectory, label="Lorentz trace")

# get fraction of k2 array
init_percent = 0 # of half width
end_percent = 0.9 # of half width
init_index = math.floor(k_as_index(kf)*init_percent)
end_index = math.floor(k_as_index(kf)*end_percent)
plt.vlines(k[end_index], -10, 10, color='black')
fractional_width = end_index-init_index
fractional_k = np.zeros(end_index - init_index)

for i in range(fractional_width):
    fractional_k[i] = k[i + init_index]

fractional_super_state_trajectory = np.zeros(fractional_width)
for i in range(fractional_width):
    fractional_super_state_trajectory[i] = super_state_trajectory[i+init_index]

def trajectory_form(x, dk):
    return -((a * x ** 2 + c) ** 2 + dk ** 2) ** 0.5

fit_params_7, pcov_7 = scipy.optimize.curve_fit(trajectory_form, fractional_k, fractional_super_state_trajectory)
print("==============================================")
print("fit_params_trajectory (dk): " + str(fit_params_7))

def trajectory_graph(k):
    return trajectory_form(k, *fit_params_7)

plt.plot(k, trajectory_graph(k), label='trajectory fit')
plt.plot(k, -E(k), label='perfect')
plt.legend()

print("residual sum squares: " + str(np.sum(np.square(trajectory_graph(fractional_k) - fractional_super_state_trajectory))))
'''