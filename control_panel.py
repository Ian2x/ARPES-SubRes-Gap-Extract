import random
##################################################
# CONTROL PANEL
##################################################

# noise_percentage = 0.02 # typically 1-3%
dk = 1 # random.uniform(0,40)# 10 # 0-40 mev
energy_conv_sigma = 2.5 # 0.0001  # 4 mev (2-5 to 10-20) 2.5?, 7.5?, and _?
min_fit_count = 25

# linewidth broadening due to the finite lifetime of photoholes
T = 5 # 22mev? (5-10mev)x

scaleup_factor = 200 * (energy_conv_sigma + T)


# 1, 2.5, 5 and 44, 54, 15

'''
peak counts at 40, 250, 1500

1 mev gap, 3 mev energy res, /50000 scaleup
10 mev gap, 15 mev energy res, 50000 scaleup (peak ~ 1000)
40 mev gap, 50 mev energy res, 30000 scaleup (peak ~ 1500)
'''