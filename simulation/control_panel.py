import random

##################################################
# SIMULATION CONTROL PANEL
##################################################

# [Gap Size]: typically 0-40 meV
dk = 15

# [Energy Resolution]: typically 2-50 meV
energy_conv_sigma = 8

# [Minimum electron count to fit]: ignore noisy data with counts lower than this value
min_fit_count = 4

# [Single-particle scattering rate]: manifests as line-width broadening; typically 5-15 meV
T = 5

# [Electron count scale-up factor]: scales count size; roughly considers broadening effects (energy_conv_sigma and T)
scaleup_factor = 2000 * (energy_conv_sigma + T)