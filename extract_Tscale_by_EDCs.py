from extract_ac_by_trajectory import *

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

# fit_data = np.zeros((len(w), len(k)))
# for i in range(len(k)):
#     ki = k[i]
#     local_scale = d6_polynomial(ki, 2.3649e+09, 1.5255e+08, -2.673779e+07, -2567193, -29517.71, 63331.38)
#     local_T = d7_polynomial(ki, -2.437345e+07, 613595, 279541, -8076.123, -1500.11, 27.15221, 11.19208)
#     temp_slice = spectrum_slice_array_SEC(w, local_scale, local_T, 14.7245282, 110.608196, -29.6103066, 0.44282100, -4.31805570, 2395.67188, -19.7310873, ki)
#     for j in range(len(w)):
#         fit_data[j][i] = temp_slice[j]
#
# im = plt.imshow(fit_data, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])  # drawing the function
# plt.colorbar(im)
# plt.show()
#
# im = plt.imshow(fit_data-Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])  # drawing the function
# plt.colorbar(im)
# plt.show()

##################################################
# Extract T and scale by fitting EDCs
##################################################

scale_trajectory = [60651.75815800848, 60260.120456892575, 61631.62239976294, 61408.72205852725, 62628.34299752422,
                    62606.81149998055, 62111.12473759114, 61438.596271462, 60725.99526057782, 61872.393203334344,
                    61680.382129755526, 61793.661486603734, 62379.53903182514, 61541.011812568126, 61180.081115610454,
                    62273.15200850579, 62000.37865828329, 61823.7147043486, 62691.691270720534, 62203.194830651664,
                    61837.461306387006, 61637.05062621418, 62404.94084354079, 62346.35666270046, 62279.72321378947,
                    61985.68547272208, 62791.90855757378, 62582.82893962801, 62342.21828785266, 61996.611940880386,
                    61911.07263639457, 61148.32752570044, 62232.94581849453, 61969.918036065894, 61635.8645258864,
                    61197.37559587275, 61204.41772705305, 61875.45348287003, 61721.58476282189, 61862.85907468815,
                    63154.040393393705, 63502.790406780645, 63603.63753696414, 63637.70423837744, 63160.06486938635,
                    63091.88494191063, 63938.28106239526, 64657.43624065157, 64720.205771168905, 64554.90372925636,
                    64336.600879296304, 64780.184807132566, 64674.34707116248, 64324.861926527694, 63931.25065085374,
                    63666.325749880656, 63344.90367665881, 63402.95781754544, 63640.02985114913, 63170.07890901893,
                    62015.25793888023, 62032.355944142335, 61848.66887741199, 61651.19994775205, 61297.85083294569,
                    61655.29793105842, 62069.47611524781, 62445.495889520564, 61860.57925617031, 61284.40695001668,
                    60730.41974352499, 60768.46485296007, 60705.00699463236, 60283.42434633815, 59228.78602388807,
                    59261.37010344283, 58434.318330972856, 58150.22970285206, 57834.480080599766, 57510.48415822221,
                    57250.91223335843, 57031.792123524814, 56534.50577130739, 56156.32399823796, 55504.916325896564,
                    53940.6202744266, 53737.300804573206, 54160.93077296758, 54197.12576063401, 52847.12467962934,
                    52360.11909260627, 52082.588671083446, 52251.170347204374, 51423.59828645866, 51326.937428674624,
                    50937.57859903936, 49403.00000414915, 49130.4352310802, 48397.97036138851, 48270.29976936863,
                    48134.37029169383, 47306.342672033854, 47359.36264837391, 45915.753077036316, 45526.14615508459,
                    44098.580305608346]

T_trajectory = [6.277783204639358, 6.337101040356318, 6.697911527354081, 6.787151754088465, 7.107378051372689,
                7.207600792102498, 7.240420839090706, 7.239549043851811, 7.216812248848585, 7.4948428387748915,
                7.536306797783325, 7.5972651105087765, 7.757697566739157, 7.680369482289821, 7.724300648748279,
                7.986597530713163, 8.018562103695512, 8.060612400076485, 8.266271641426817, 8.239078118706557,
                8.222180615313624, 8.240643291729993, 8.396468995566675, 8.497929653653784, 8.59254005896753,
                8.645400244950348, 8.808130353917964, 8.83625571816249, 8.849078393752812, 8.88622657038399,
                8.948422439901538, 8.861722600394915, 9.104500495133694, 9.157149252906514, 9.187533663432028,
                9.221237782403906, 9.28638444347058, 9.471305153847084, 9.511899243118057, 9.628268110923482,
                9.9628059478444, 10.066529575079462, 10.190003895820391, 10.29367614765471, 10.31036062444199,
                10.39621903978088, 10.620273941745312, 10.828910071278733, 10.92641657772972, 11.01165677434713,
                11.081204688004663, 11.255719428552196, 11.280758186240575, 11.288224250074228, 11.277703698576367,
                11.267769174164467, 11.24045659269359, 11.314361238436847, 11.364687486178038, 11.267287600664428,
                11.048325445142257, 11.105301248465908, 11.130828520989498, 11.153069752340814, 11.142985318332416,
                11.221101565915566, 11.275353134202941, 11.32113540607598, 11.189713589593099, 11.055140630592296,
                10.922773704505047, 10.963629216675525, 10.977273622901812, 10.958204850351951, 10.815872788480961,
                10.77635056100904, 10.47354261172309, 10.354107947362785, 10.224284196640722, 10.156570286372272,
                10.082050685202198, 10.006107909421944, 9.8710845633477, 9.81409691730574, 9.758019878408886,
                9.5059904031601, 9.418904740384605, 9.45316739829045, 9.410337324647562, 9.130613956086142,
                9.00601023942602, 8.921171665415102, 8.920528658786715, 8.715659764808576, 8.75358713056615,
                8.733863534195736, 8.454727448065901, 8.379453634979342, 8.195133227232077, 8.103834172705135,
                8.00808505585328, 7.845294803435986, 7.807926612008263, 7.482003241618352, 7.3695810349640425,
                7.036793918737691]
'''
# scale_fit_d4_params, scale_fit_d4_pcov = scipy.optimize.curve_fit(d4_polynomial, k, scale_trajectory)
# scale_fit_d5_params, scale_fit_d5_pcov = scipy.optimize.curve_fit(d5_polynomial, k, scale_trajectory)
scale_fit_d6_params, scale_fit_d6_pcov = scipy.optimize.curve_fit(d6_polynomial, k, scale_trajectory)
# scale_fit_d10_params, scale_fit_d10_pcov = scipy.optimize.curve_fit(d10_polynomial, k, scale_trajectory)

pars = lmfit.Parameters()
pars.add('a', value=0)
pars.add('b', value=0)
pars.add('c', value=0)
pars.add('d', value=0)
pars.add('e', value=0)
pars.add('f', value=0)




def residual(p):
    return d6_polynomial(k, p['a'], p['b'], p['c'], p['d'], p['e'], p['f']) - scale_trajectory


mini = lmfit.Minimizer(residual, pars, nan_policy='propagate', calc_covar=True)
result = mini.minimize(method='least_squares')
print(lmfit.fit_report(result))

fit6 = d6_polynomial(k, *scale_fit_d6_params)

# print(manualFTest(scale_trajectory, fit4, 4, fit5, 5, np.ones(len(scale_trajectory)), len(scale_trajectory)))
print(scale_fit_d6_params)
plt.plot(k, scale_trajectory, label='data')
plt.plot(k, fit6, label='6')
plt.legend()
plt.show()

plt.plot(k, scale_trajectory, label='data')
lmfit_answer = d6_polynomial(k, result.params.get('a').value, result.params.get('b').value, result.params.get('c').value,
                          result.params.get('d').value, result.params.get('e').value, result.params.get('f').value)
plt.plot(k, lmfit_answer, label='6 new')
plt.show()
print(np.sum(np.square(fit6-scale_trajectory)))
print(np.sum(np.square(lmfit_answer-scale_trajectory)))
quit()

# scale poly 5 --> same, 55665169.27688192
# scale poly 6 --> least_squares, 45948611.54764917
# scale poly 7 --> same

print(scale_fit_d4_params)
plt.plot(k, scale_trajectory, label='data')
plt.plot(k, d4_polynomial(k, *scale_fit_d4_params), label='4')
plt.legend()
plt.show()

print(scale_fit_d6_params)
plt.plot(k, scale_trajectory, label='data')
plt.plot(k, d6_polynomial(k, *scale_fit_d6_params), label='6')
plt.legend()
plt.show()

print(scale_fit_d7_params)
plt.plot(k, scale_trajectory, label='data')
plt.plot(k, d7_polynomial(k, *scale_fit_d7_params), label='7')
plt.legend()
plt.show()

T_fit_params, T_fit_pcov = scipy.optimize.curve_fit(d7_polynomial, k, T_trajectory)
print(T_fit_params)
plt.plot(k, T_trajectory, label='data2')
plt.plot(k, d7_polynomial(k, *T_fit_params))
plt.legend()
plt.show()


scale_trajectory = np.zeros(z_width)
T_trajectory = np.zeros(z_width)
scale_0, T_0, dk_0, a_0 = np.ones(4)
c_0 = 0
temp_list = [45,46]
for i in range(z_width): # range(z_width):
    temp_w, temp_slice, temp_fitting_sigma, temp_points_in_fit, temp_fit_start_index, temp_fit_end_index = EDC_prep(i, Z, w)

    temp_params, temp_pcov = scipy.optimize.curve_fit(partial(spectrum_slice_array, a=initial_a_estimate, c=initial_c_estimate, fixed_k=k[i]), temp_w, temp_slice,
                                                      bounds=(
                                                          [1 / ONE_BILLION, 0, 0],
                                                          [ONE_BILLION, 75, 75]),
                                                      p0=[scale_0, T_0, dk_0],
                                                      sigma=temp_fitting_sigma)
    scale_trajectory[i] = temp_params[0]
    T_trajectory[i] = temp_params[1]
    scale_0, T_0, dk_0= temp_params
    # plt.plot(temp_w, temp_slice)
    # plt.plot(temp_w, spectrum_slice_array(temp_w, *temp_params, initial_a_estimate, initial_c_estimate, k[i]))
    # plt.show()
    print(i)
    # print(temp_params)
    # print(temp_pcov)
scale_trajectory = [i for i in scale_trajectory if i != 0]
T_trajectory = [i for i in T_trajectory if i != 0]
'''
plt.title("scale")
plt.plot(k, scale_trajectory)
plt.plot(k,
         d6_polynomial(k, 2.3649e+09, 1.5255e+08, -2.67377948e+07, -2.56719251e+06, -2.95177140e+04,
                       6.33313795e+04))
plt.show()
plt.title("T")
plt.plot(k, T_trajectory)
plt.plot(k, d7_polynomial(k, -2.43734497e+07, 6.13594963e+05, 2.79541030e+05, -8.07612271e+03, -1.50010981e+03, 2.71522060e+01, 1.11920804e+01))
plt.show()
# print(scale_trajectory)
# print(T_trajectory)