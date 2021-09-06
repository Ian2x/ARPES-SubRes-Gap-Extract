from gold_reference import *
'''
real_data_file = open(r"/Users/ianhu/Downloads/OD50#9_0311.txt", "r")

real_data = np.zeros((201,700))

k_real = np.linspace(-16.48000, 15.47429, num=700) # in degrees
w_real = np.linspace(13970.0, 14170.0, num=201) # in mev

# get to [Data 1] line
while real_data_file.readline() != '[Data 1]\n':
    pass

for i in range(201):
    temp_k_slice = real_data_file.readline()
    temp_split_array = temp_k_slice.split()
    for j in range(700):
        real_data[201-i-1][j] = temp_split_array[j+1] # ignore first

# plot real data reference
plt.title("Raw real data")
im = plt.imshow(real_data, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k_real), max(k_real), min(w_real), max(w_real)])  # drawing the function
plt.colorbar(im)
plt.show()

# 0189 --> 0233
real_data_file_2 = open(r"/Users/ianhu/Desktop/ARPES Shared Data/X20141210_near_node/OD50_0189_nL.dat", "r")

temp = real_data_file_2.readline() # skip blank starting line
temp = real_data_file_2.readline() # energy?
temp_split = temp.split()

real_data_2 = np.zeros((401,690))
temp_k = np.zeros(690)
temp_w = np.zeros(401)

for i in range(401):
    temp_w[i] = temp_split[i]

temp = real_data_file_2.readline() # empty 0.0164694505526385
for i in range(690):
    temp = real_data_file_2.readline()
    temp_split = temp.split()
    # temp_k[i] = temp_split[0]
    temp_k[690-i-1] = -float(temp_split[0]) # flip to positive
    for j in range(401):
        # real_data_2[401-j-1][690-i-1] = temp_split[j+1]
        real_data_2[401-j-1][i] = temp_split[j+1] # flip to positive


exp_height = 250
exp_width = 100
height_offset = 20
width_offset = 260

Z = np.zeros((exp_height, exp_width))
for i in range(exp_height):
    for j in range(exp_width):
        Z[i][j] = real_data_2[i+height_offset][j+width_offset]

plt.title("Raw real data")
im = plt.imshow(real_data_2, cmap=plt.cm.RdBu, aspect='auto', extent=[min(temp_k), max(temp_k), min(temp_w), max(temp_w)])  # drawing the function
plt.colorbar(im)
plt.show()


Z = np.multiply(Z, 5000)
Z = np.around(Z)


plt.title("Reduced window")
im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[temp_k[690-width_offset-1], temp_k[690-width_offset-exp_width-1], temp_w[401-height_offset-exp_height-1], temp_w[401-height_offset-1]])
plt.colorbar(im)
plt.show()
'''