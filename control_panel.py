import random

##################################################
# CONTROL PANEL
##################################################

# [Gap Size]: typically 0-40 meV
dk = 5

# [Energy Resolution]: typically 2-50 meV
energy_conv_sigma = 2

# [Minimum electron count to fit]: ignore noisy data with counts lower than this value
min_fit_count = 4

# [Single-particle scattering rate]: manifests as line-width broadening; typically 5-15 meV
T = 5

# [Electron count scale-up factor]: scales count size; roughly considers broadening effects (energy_conv_sigma and T)
scaleup_factor = 200 * (energy_conv_sigma + T)

# 1, 2.5, 5 and 44, 54, 15

'''
peak counts at 40, 250, 1500

1 mev gap, 3 mev energy res, /50000 scaleup
10 mev gap, 15 mev energy res, 50000 scaleup (peak ~ 1000)
40 mev gap, 50 mev energy res, 30000 scaleup (peak ~ 1500)
'''
