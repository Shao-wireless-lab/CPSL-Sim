import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import matplotlib.colors as mcolors
from datetime import datetime


matplotlib.use('TkAgg')

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

# Load the .mat files
#mat_data1 = scipy.io.loadmat("/home/ece213/CPSL-Sim/cuas/plume_data/Plume-C-Data-Height-2.mat")
#mat_data2 = scipy.io.loadmat("/home/ece213/CPSL-Sim/cuas/plume_data/Plume-C-Data-Height-5.mat")
mat_data1 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plumedata/Plume-C-Data-Height-2_new.mat")
mat_data2 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plumedata/Plume-C-Data-Height-5_new.mat")

# Emitter position in (x, y) format as per Cartesian coordinates
# TODO: import value from config
emitter_position = [40, 30]

# Extract x, y values for Matplotlib plotting
emitter_x, emitter_y = emitter_position[0], emitter_position[1]

C1 = mat_data1['C']  # Replace 'C' with the actual variable name if different
C2 = mat_data2['C']  # Replace 'C' with the actual variable name if different

# Reshape the data to (6000, 100, 100)
num_timesteps = 6000
grid_size = 100
C1 = C1.reshape((num_timesteps, grid_size, grid_size))
C2 = C2.reshape((num_timesteps, grid_size, grid_size))

# Transpose the matrices to match the correct positioning
C1 = np.transpose(C1, (0, 2, 1))
C2 = np.transpose(C2, (0, 2, 1))

# Define threshold
threshold = 1.990

# Create masked arrays: values <= threshold will be transparent (white)
masked_C1 = np.ma.masked_less_equal(C1, threshold)
masked_C2 = np.ma.masked_less_equal(C2, threshold)

# Create a colormap: White for below threshold, Green for above
colors = [(1, 1, 1, 0), (0, 0.5, 0, 1)]  # (R, G, B, Alpha)
custom_cmap = mcolors.LinearSegmentedColormap.from_list("white_to_green", colors, N=256)

# Set up figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


# Determine vmax dynamically from the data
vmax = max(np.max(C1), np.max(C2))  # Get the highest value across both datasets
vmin = 0  # Set minimum to 0 for better visualization

cax1 = ax1.imshow(masked_C1[0], cmap=custom_cmap, interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)
cax2 = ax2.imshow(masked_C2[0], cmap=custom_cmap, interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)

plt.colorbar(cax1, ax=ax1)
plt.colorbar(cax2, ax=ax2)

ax1.set_title('Dataset 1: Plume Height 2')
ax2.set_title('Dataset 2: Plume Height 5')

# Draw a red dot at emitter's position
red_dot1, = ax1.plot(emitter_x, emitter_y, 'ro', markersize=5)
red_dot2, = ax2.plot(emitter_x, emitter_y, 'ro', markersize=5)

# Function to update animation
def update(frame):
    cax1.set_data(masked_C1[frame])
    cax2.set_data(masked_C2[frame])
    ax1.set_title(f'Plume Height 2: Timestep {frame+1}')
    ax2.set_title(f'Plume Height 5: Timestep {frame+1}')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_timesteps, interval=50, repeat=True)

# Choose action: "show" or "save"
action = "show"  # Change to "show" if you prefer to display the animation directly

if action == "show":
    plt.show()
elif action == "save":
    video_filename = f"concentration_{timestamp}.mp4"
    print(f"Animation is starting to save as {video_filename}")
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save(video_filename, writer=writervideo, dpi=200)
    print(f"Animation saved as {video_filename}")

