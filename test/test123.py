'''
from datetime import datetime


timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")\

print(timestamp)

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the .mat files
mat_data1 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plume_data/Plume-C-Data-Height-2.mat")
mat_data2 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plume_data/Plume-C-Data-Height-5.mat")

# Emitter position in (x, y) format
emitter_position = [40, 30]

# Extract data
C1 = mat_data1['C']
C2 = mat_data2['C']

# Reshape data
num_timesteps = 6000
grid_size = 100
C1 = C1.reshape((num_timesteps, grid_size, grid_size))
C2 = C2.reshape((num_timesteps, grid_size, grid_size))

# Transpose to match coordinate system
C1 = np.transpose(C1, (0, 2, 1))
C2 = np.transpose(C2, (0, 2, 1))

# Define threshold
threshold = 1.990

# Mask values below threshold
masked_C1 = np.ma.masked_less_equal(C1, threshold)
masked_C2 = np.ma.masked_less_equal(C2, threshold)

# Create a colormap
colors = [(1, 1, 1, 0), (0, 0.5, 0, 1)]  # (R, G, B, Alpha)
custom_cmap = mcolors.LinearSegmentedColormap.from_list("white_to_green", colors, N=256)

# Determine vmax dynamically
vmax = max(np.max(C1), np.max(C2))
vmin = 0

# Time steps to visualize
timesteps_to_plot = [0, num_timesteps // 2, num_timesteps - 1]  # First, middle, last

for t in timesteps_to_plot:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    cax1 = ax1.imshow(masked_C1[t], cmap=custom_cmap, interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)
    cax2 = ax2.imshow(masked_C2[t], cmap=custom_cmap, interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)

    plt.colorbar(cax1, ax=ax1)
    plt.colorbar(cax2, ax=ax2)

    ax1.set_title(f'Plume Height 2: Timestep {t}')
    ax2.set_title(f'Plume Height 5: Timestep {t}')

    # Draw emitter position
    ax1.plot(emitter_position[0], emitter_position[1], 'ro', markersize=5)
    ax2.plot(emitter_position[0], emitter_position[1], 'ro', markersize=5)

    #plt.savefig(f"plume_distribution_t{t}.png", dpi=200)
    #print(f"Saved: plume_distribution_t{t}.png")

plt.show()
'''
'''
import scipy.io

# Build paths dynamically
wind_speed_5 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plume_data/Plume-Wind-Data-Height-5.mat")
Conc_5 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plume_data/Plume-C-Data-Height-5.mat")

wind_speed_2 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plume_data/Plume-Wind-Data-Height-2.mat")
Conc_2 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plume_data/Plume-C-Data-Height-2.mat")

ws_Height_2 = wind_speed_2["ws"]
rho_Height_2 = Conc_2["C"]

ws_Height_5 = wind_speed_5["ws"]
rho_Height_5 = Conc_5["C"]

print(f"shape for windspeed height2: {ws_Height_2.shape}")
print(f"shape for concentration height2: {rho_Height_2.shape}")
print(f"shape for windspeed height5: {ws_Height_5.shape}")
print(f"shape for concentration height5: {rho_Height_5.shape}")

# Modify the data directly
wind_speed_2["ws"] = wind_speed_2["ws"][-320000:, :]
Conc_2["C"] = Conc_2["C"][-320000:, :]
wind_speed_5["ws"] = wind_speed_5["ws"][-320000:, :]
Conc_5["C"] = Conc_5["C"][-320000:, :]

# Print new shapes to confirm
print(f"Modified shape for windspeed height2: {wind_speed_2['ws'].shape}")
print(f"Modified shape for concentration height2: {Conc_2['C'].shape}")
print(f"Modified shape for windspeed height5: {wind_speed_5['ws'].shape}")
print(f"Modified shape for concentration height5: {Conc_5['C'].shape}")
'''


import scipy.io

mat_data1 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/plume_data/Plume-Wind-Data-Height-2_G_3.mat")

# print out keys
print(mat_data1.keys())

