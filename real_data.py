from gold_reference import *

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