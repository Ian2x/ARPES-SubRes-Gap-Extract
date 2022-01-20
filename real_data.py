from simulation.heat_map import *
'''
shared_drive_data_file = open(r"/Users/ianhu/Downloads/OD50#9_0311.txt", "r")
shared_drive_data = np.zeros((201, 700))
k = np.linspace(-16.48000, 15.47429, num=700) # in degrees
w = np.linspace(13970.0, 14170.0, num=201) # in mev
# get to [Data 1] line
while shared_drive_data_file.readline() != '[Data 1]\n':
    pass
for i in range(201):
    temp_k_slice = shared_drive_data_file.readline()
    temp_split_array = temp_k_slice.split()
    for j in range(700):
        shared_drive_data[201 - i - 1][j] = temp_split_array[j + 1] # ignore first
# plot real data reference
plt.title("Raw real data")
im = plt.imshow(shared_drive_data, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])  # drawing the function
plt.colorbar(im)
plt.show()
'''

# 0189 --> 0233
# /Users/ianhu/Desktop/ARPES Shared Data/X20141210_near_node/OD50_0189_nL.dat
# 0289 --> 0333
# /Users/ianhu/Desktop/ARPES Shared Data/X20141210_far_off_node/OD50_0289_nL.dat

# Eugen_data_file = open(r"/Users/ianhu/Documents/ARPES/ARPES Shared Data/X20141210_far_off_node/OD50_0333_nL.dat", "r")
Eugen_data_file = open(r"OD50_0333_nL.dat", "r")

Eugen_data_file.readline() # skip blank starting line
temp = Eugen_data_file.readline() # energy?
temp_split = temp.split()

w_dim = 201 # 401 for near node # 201 for far off node
k_dim = 695 # 690 for near node # 695 for far off node

Eugen_data = np.zeros((w_dim, k_dim))
k = np.zeros(k_dim)
w = np.zeros(w_dim)
for i in range(w_dim):
    w[i] = float(temp_split[i]) * 1000
w = np.flip(w)

Eugen_data_file.readline() # empty 0.0164694505526385 / 0.515261371488587
Eugen_data_file.readline() # unfilled 0.513745070571566 (FOR FAR OFF NODE ONLY)
Eugen_data_file.readline() # unfilled 0.512228769654545 (FOR FAR OFF NODE ONLY)

for i in range(k_dim):
    temp = Eugen_data_file.readline()
    temp_split = temp.split()
    k[i] = float(temp_split[0]) # flip to positive --> removed negative
    for j in range(w_dim):
        Eugen_data[w_dim - j - 1][k_dim - i - 1] = temp_split[j + 1] # fill in opposite
k = np.flip(k)
plt.title("Raw Eugen data")
im = plt.imshow(Eugen_data, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)])  # drawing the function
plt.colorbar(im)
plt.show()
# ===== Reduce window ===== #

# 250, 100, 20, 260 for near node
# 60, 100, 30, 340 for far off node
z_height = 70 # from 110
z_width = 106
height_offset = 40 # from 0
width_offset = 311

temp_k = np.zeros(z_width)
temp_w = np.zeros(z_height)

Z = np.zeros((z_height, z_width))

for i in range(z_height):
    temp_w[i] = w[i + height_offset]
    for j in range(z_width):
        Z[i][j] = Eugen_data[i + height_offset][j + width_offset]
        temp_k[j] = k[j + width_offset]

Z = np.multiply(Z, 5000)
Z = np.around(Z)

# Overwrite original k,w with reduced window k,w
k = temp_k
w = temp_w

energy_conv_sigma = 8 / 2.35482004503

inv_Z = np.array([list(i) for i in zip(*Z)])
print(w)