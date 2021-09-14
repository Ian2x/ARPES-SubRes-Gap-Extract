from real_data import *

##################################################
# EXTRACT KF FROM MDC AT EF
##################################################

fermi_mdc = np.zeros(z_width)
ef_index = w_as_index(0, w)

print(ef_index)
print(w)

for i in range(z_width):
    fermi_mdc[i] = Z[z_height-ef_index-1][i]
    print(Z[ef_index][i])


# Dirac delta function
def dirac_delta(x, a, scale, hori_shift, vert_shift):
    return scale * math.e ** (- (((x - hori_shift) / a) ** 2)) + vert_shift


dirac_delta_vectorized = np.vectorize(dirac_delta)

kf_extract_params, kf_extract_cov = scipy.optimize.curve_fit(dirac_delta, k, fermi_mdc)
extracted_kf = kf_extract_params[2]
kf = extracted_kf

extracted_c = c
extracted_a = -extracted_c / (extracted_kf ** 2)

plt.plot(k, dirac_delta(k, *kf_extract_params), label='Fit')
plt.plot(k, fermi_mdc)
plt.legend()
plt.show()

print("extracted params: ", kf_extract_params)
print("extracted a: ", extracted_a)