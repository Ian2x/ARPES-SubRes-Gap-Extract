##################################################
# EUGEN DATA GENERATION
##################################################

# counts as int
'''
dk_selection = [
    1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2,
    8, 8, 8, 10, 10, 10, 12, 12, 12,
    36, 36, 36, 40, 40, 40, 44, 44, 44]
energy_conv_sigma_selection = [
    2.5, 3, 3.5, 2.5, 3, 3.5, 2.5, 3, 3.5,
    13, 15, 17, 13, 15, 17, 13, 15, 17,
    46, 50, 54, 46, 50, 54, 46, 50, 54]
T_selection = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
scaleup_factor_selection = [200, 400, 700, 1000, 2000, 4000, 7000, 10000, 15000, 20000]
'''
dk_selection = [
    1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2,
    8, 8, 8, 10, 10, 10, 12, 12, 12,
    36, 36, 36, 40, 40, 40, 44, 44, 44]
energy_conv_sigma_selection = [
    2.5, 3, 3.5, 2.5, 3, 3.5, 2.5, 3, 3.5,
    13, 15, 17, 13, 15, 17, 13, 15, 17,
    46, 50, 54, 46, 50, 54, 46, 50, 54]
T_selection = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
scaleup_factor_selection = [200, 400, 700, 1000, 2000, 4000, 7000, 10000, 15000, 20000]

progress_count = 0
for x in range(len(dk_selection)):
    for y in range(len(T_selection)):
        for z in range(len(scaleup_factor_selection)):

            progress_count+=1
            print(str(progress_count) + "/" + str(len(dk_selection)*len(T_selection)*len(scaleup_factor_selection)))

            dk = dk_selection[x]
            energy_conv_sigma = energy_conv_sigma_selection[x]
            T = T_selection[y]
            scaleup_factor = scaleup_factor_selection[z] * (energy_conv_sigma + T)


            Z = I(X, Y)
            add_noise(Z)
            Z = Z.astype(int)

            if x==0 and y==0:
                # plot 10 times
                # plt.cm.RdBu
                im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)],
                                origin='lower')  # drawing the function
                plt.colorbar(im)
                plt.show()

            file = open(r"/Users/ianhu/Documents/ARPES/Preliminary Coarse-Graining/" + "dk=" + str(dk) + ",ER=" + str(
                energy_conv_sigma) + ",T=" + str(T) + ",CF=" + str(round(scaleup_factor / (energy_conv_sigma + T), 2)) + ".txt", "w+")
            file.write("dk [meV]=" + str(dk) + '\n')
            file.write("energy res [meV]=" + str(energy_conv_sigma) + '\n')
            file.write("T [meV]=" + str(T) + '\n')
            file.write("Count factor [energy res+T]=" + str(scaleup_factor / (energy_conv_sigma + T)) + '\n\n')

            # print momentum [A^-1]
            file.write("Width name=Momentum [A^-1]\n")
            file.write("Width size=" + str(z_width) + '\n')
            for i in range(k.size):
                file.write(str(round(k[i], 3)) + '\t')
            file.write('\n\n')

            # print energy [meV]
            file.write("Height name=Energy [meV]\n")
            file.write("Height size=" + str(z_height) + '\n')
            for i in range(w.size):
                file.write(str(w[i]) + '\t')
            file.write('\n\n')

            for i in range(z_width):
                # print k
                file.write(str(round(k[i], 3)) + '\t')
                for j in range(z_height):
                    # print Z
                    file.write(str(Z[j][i]) + '\t')
                file.write('\n')

            file.close()
quit()
