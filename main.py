from heat_map_setup import *

# from random import randrange

from functools import partial
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate

'''
dk_selection = [
    1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2,
    8, 8, 8, 10, 10, 10, 12, 12, 12,
    36, 36, 36, 40, 40, 40, 44, 44, 44]
energy_conv_sigma_selection = [
    2.5, 3, 3.5, 2.5, 3, 3.5, 2.5, 3, 3.5,
    13, 15, 17, 13, 15, 17, 13, 15, 17,
    46, 50, 54, 46, 50, 54, 46, 50, 54]
'''

dk_selection = [
    36, 36, 36, 40, 40, 40, 44, 44, 44]
energy_conv_sigma_selection = [
    46, 50, 54, 46, 50, 54, 46, 50, 54]
T_selection = [5, 7, 9, 11, 13, 15]
prescaleup_selection = [200, 400, 700, 1000, 2000, 4000, 7000, 10000, 15000, 20000]


progress_count = 0
temp_skip = 4
for x in range(len(dk_selection)):
    for y in range(len(T_selection)):
        for z in range(len(prescaleup_selection)):

            if temp_skip > 0:
                temp_skip-=1
                continue

            # Print progress
            progress_count+=1
            print(str(progress_count) + "/" + str(len(dk_selection) * len(T_selection) * len(prescaleup_selection)))

            dk = dk_selection[x]
            energy_conv_sigma = energy_conv_sigma_selection[x]
            T = T_selection[y]
            scaleup_factor = prescaleup_selection[z] * (energy_conv_sigma + T)

            # [Minimum electron count to fit]: ignore noisy data with counts lower than this value
            min_fit_count = scaleup_factor / 2500

            ##################################################
            # DISPERSION SETUP
            ##################################################

            # [Lattice Constant]
            lattice_a = 4  # typically 3.6-4.2 Angstrom

            # [Brillouin momentum]
            brillouin_k = math.pi / lattice_a  # angstrom || # hbar * math.pi / lattice_a # g*m/s

            # [Fermi momentum]
            kf = 0.5092958179 * brillouin_k  # 0.5092958179
            # Overwrite Fermi Momentum to exactly 0.4 (A^-1)
            kf = 0.4

            # Normal-state Dispersion Variables (ak^2+c)
            c = min(-50 * dk, -50)  # typically -1000 to -10,000 mev
            a = -c / (kf ** 2)
            # ------------------------------------------------
            # [Energy pixel step size]
            min_w = round(-4.5*(dk+energy_conv_sigma))
            max_w = round(3.5*(dk+energy_conv_sigma))
            w_step = (max_w-min_w) / 135 # 135 steps --> 160 steps
            if w_step < 1:
                w_step = 1
            elif w_step > 4:
                w_step = 4

            # [Energy detectors array]
            w = np.arange(min_w, max_w, w_step)


            # Convert w in meV to corresponding index
            def w_as_index(input_w):
                return int(round((input_w - min(w)) / (max(w) - min(w)) * (w.size - 1)))


            # [Angle uncertainty]: typically 0.1-0.2 degrees, using 0.045 degrees here
            d_theta = 0.045 * math.pi / 180

            # [Momentum pixel step size]
            k_step = (1 / hbar) * math.sqrt(
                2 * mass_electron / speed_light / speed_light * (6176.5840329647)) * d_theta / (
                             10 ** 10)

            # [Momentum detectors array]
            k = np.arange(kf - 0.04 * kf, kf + 0.04 * kf, k_step)
            # print("k_step: " + str(k_step) + " | mink: " + str(min(k)) + " | maxk: " + str(max(k)) + " | #steps: " + str(k.size))


            # Convert k in A^-1 to corresponding index
            def k_as_index(input_k):
                return int(round((input_k - min(k)) / (max(k) - min(k)) * (k.size - 1)))


            ##################################################
            # HEAT MAP
            ##################################################

            # k and w values for plotting
            X, Y = np.meshgrid(k, w)

            # The Spectrum
            Z = I(X, Y, a, c, dk, T, scaleup_factor, energy_conv_sigma)
            # print("Z.size:", Z.shape, "\n")
            add_noise(Z)

            z_width = Z[0].size
            z_height = int(Z.size / z_width)
            kf_index = k_as_index(kf)

            ##################################################
            # GAP EXTRACTION
            ##################################################

            # Integrand of energy convolution integral
            def energy_conv_integrand(integration_w, fixed_w, T, dk, a, c, fixed_k):
                return A_BCS(fixed_k, integration_w, a, c, dk, T) * R(math.fabs(integration_w - fixed_w),
                                                                      energy_conv_sigma) * n(
                    integration_w)


            # EDC slice function
            def spectrum_slice_array(w_array, scale, T, dk, a, c, fixed_k):
                return_array = np.zeros(w_array.size)
                for i in range(w_array.size):
                    return_array[i] = scale * \
                                      scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250,
                                                           args=(w_array[i], T, dk, a, c, fixed_k))[0]
                return return_array


            left_shift_mult = 2
            fit_start_k = 0
            if a != 0:
                fit_start_k = math.sqrt((-left_shift_mult * dk - c) / a)
            short_k = np.linspace(max(0, k_as_index(fit_start_k)), k_as_index(kf), 6)
            short_k = short_k.astype(int)
            short_k = np.unique(short_k)
            short_k = np.flip(short_k, 0)  # try to fit starting from fermi slice
            # print(short_k)
            curr_index = 0

            last_dk = 1
            last_scale = scaleup_factor / 10
            last_T = 1

            # set up multiple subplots
            num_plots = short_k.size + 1
            plt.figure(figsize=(6 * num_plots, 6), dpi=120)
            first_plot = plt.subplot(1, num_plots, 1)
            first_plot.set_title("Spectrum (dk=" + str(dk) + ")")
            im = first_plot.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)],
                                   origin='lower')  # drawing the function
            # im = first_plot.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[temp_k[690-width_offset-1], temp_k[690-width_offset-exp_width-1], temp_w[401-height_offset-exp_height-1], temp_w[401-height_offset-1]])

            plt.colorbar(im)
            plt.xlabel('k ($A^{-1}$)')
            plt.ylabel('w (mev)')
            plt.show()
            # /Users/ianhu/PycharmProjects/ARPES-SubRes-Gap-Extract
            # np.savetxt("/Users/ianhu/Documents/ARPES CNN/Dataset 1 - c=-1000, sigma=15/"+str(round(dk,6))+".csv", Z, delimiter=",")

            plt_index = 2  # first plot is for spectrum

            dk_guesses = []
            dk_errors = []
            redchis = []


            for slice_k_index in short_k:
                # Norman_subplot = plt.subplot(1, num_plots, plt_index)
                plt_index += 1
                # print('==============================================')
                # print('slice_k_index: ' + str(slice_k_index) + ' (' + str(k[slice_k_index]) + ')')
                # print('progress: ' + str(curr_index / short_k.size))
                EDC = np.zeros(z_height)
                curr_k = k[slice_k_index]

                # IGNORE NOISY DATA
                fit_start_index = -1
                fit_end_index = -1

                for i in range(z_height):
                    EDC[i] = Z[i][slice_k_index]
                    if fit_start_index == -1:
                        if EDC[i] >= min_fit_count:
                            fit_start_index = i
                    if EDC[i] >= min_fit_count:
                        fit_end_index = i
                print(fit_start_index)
                print(fit_end_index)

                # SUFFICIENT ROOM FOR ENERGY CONV
                min_indexes_from_edge = 3 * energy_conv_sigma / w_step
                fit_start_index = int(max(fit_start_index, round(min_indexes_from_edge)))
                fit_end_index = int(min(fit_end_index, round(z_height - 1 - min_indexes_from_edge)))
                points_in_fit = fit_end_index - fit_start_index + 1  # include end point
                print(fit_start_index)
                print(fit_end_index)

                # LOW NOISE SLICE CREATION
                low_noise_slice = np.zeros(points_in_fit)
                low_noise_w = np.zeros(points_in_fit)
                for i in range(points_in_fit):
                    low_noise_slice[i] = Z[i + fit_start_index][slice_k_index]
                    low_noise_w[i] = w[i + fit_start_index]

                # FUNCTION TO FIT
                fit_func_w_scale_T_dk = partial(spectrum_slice_array, a=a, c=c, fixed_k=curr_k)

                # ===== ===== ===== SCIPY ===== ===== =====

                fitting_sigma = low_noise_slice
                for i in range(len(fitting_sigma)):
                    if fitting_sigma[i] < 1:
                        fitting_sigma[i] = 1

                scipy_full_params, scipy_full_pcov = scipy.optimize.curve_fit(fit_func_w_scale_T_dk, low_noise_w,
                                                                              low_noise_slice,
                                                                              p0=[last_scale, last_T, last_dk],
                                                                              maxfev=2000, bounds=(
                    [scaleup_factor / 10, 0., 0.], [scaleup_factor * 10, 50., 50.]), sigma=np.sqrt(low_noise_slice))

                last_scale = scipy_full_params[0]
                last_T = scipy_full_params[1]
                last_dk = scipy_full_params[2]

                # print("scipy full params: ", scipy_full_params)
                scipy_std_err = math.sqrt(scipy_full_pcov[2][2] / math.sqrt(points_in_fit))
                # print("dk stderr +/- : " + str(scipy_std_err) + " (" + str(100 * scipy_std_err / last_dk) + "%)")
                # print("scipy full pcov: \n", scipy_full_pcov)

                # SCIPY STATISTICAL RESULTS
                DOF = points_in_fit - 3
                # print("DOF: ", points_in_fit - 3)
                redchi = manualRedChi( \
                    low_noise_slice, fit_func_w_scale_T_dk(low_noise_w, *scipy_full_params), \
                    low_noise_slice, points_in_fit - 3)
                # print("scipy redchi:", redchi)

                '''
                # SCIPY PLOT
                plt.plot(w, fit_func_w_scale_T_dk(w, *scipy_full_params), label='Fitted curve')
                plt.plot(w, fit_func_w_scale_T_dk(w, scaleup_factor / w_step, T, dk), label='Perfect fit')

                # DATA/REFERENCE PLOTS
                plt.plot(w, EDC, label='Data')

                # plot
                Norman_subplot.set_title(
                    "k ($A^{-1}$): " + str(round(k[slice_k_index], 3)) + " | dk estimate:" + str(round(last_dk, 2)))
                plt.vlines(w[fit_start_index], 0, min_fit_count, color='black')
                plt.vlines(w[fit_end_index], 0, min_fit_count, color='black')
                plt.xlabel('w (mev)')
                plt.ylabel('counts')
                plt.legend()
                
                curr_index += 1

                if slice_k_index == 16:
                    Norman_subplot.set_title(
                        "k ($A^{-1}$): " + str(round(k[slice_k_index], 3)) + "(kf) | dk estimate:" + str(
                            round(last_dk, 2)), color='olive')
                '''
                # EASY_PRINT_ARRAY

                dk_guesses.append(last_dk)
                dk_errors.append(scipy_std_err)
                redchis.append(redchi)

                # need to print each EDC (dk guess, dk stderr, redchi)
                # a, c, energy res, T, count factor
                # also include true dk

            ##################################################
            # SAVE TO FILES
            ##################################################
            '''
            if randrange(1)==0:
                im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)],
                                origin='lower')  # drawing the function
                plt.colorbar(im)
                plt.show()
            '''
            # /Users/ianhu/Documents/ARPES/Low SNR Systematic Error/
            # Low SNR Systematic Error/

            script_dir = os.path.dirname(__file__)
            rel_path = "Low_SNR_Systematic_Error/" + "dk=" + str(dk) + ",ER=" + str(
                energy_conv_sigma) + ",T=" + str(T) + ",CF=" + str(round(scaleup_factor / (energy_conv_sigma + T), 2)) + ".txt"
            abs_file_path = os.path.join(script_dir, rel_path)

            file = open(abs_file_path, "w+")

            file.write("True dk [meV]=" + str(dk) + '\n\n=====\n\n')
            file.write("dispersion 'a'=" + str(a) + '\n')
            file.write("dispersion 'c'=" + str(c) + '\n')
            file.write("energy res [meV]=" + str(energy_conv_sigma) + '\n')
            file.write("T [meV]=" + str(T) + '\n')
            file.write("Count factor [energy res+T]=" + str(scaleup_factor / (energy_conv_sigma + T)) + '\n\n=====\n\n')

            for i in range(len(dk_guesses)):
                file.write("momentum=" + str(k[short_k[i]]) + '\n')
                file.write("dk guess=" + str(dk_guesses[i]) + '\n')
                file.write("dk stderr=" + str(dk_errors[i]) + '\n')
                file.write("reduced chi value=" + str(redchis[i]) + '\n\n')

            file.close()
quit()
