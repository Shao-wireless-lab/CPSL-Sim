import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from datetime import datetime

matplotlib.use('TkAgg')

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

# Load .mat files
mat_data1 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plumedata/Plume-C-Data-Height-2_new.mat")
mat_data2 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plumedata/Plume-C-Data-Height-5_new.mat")

# Extract concentration data
C1 = mat_data1['C']
C2 = mat_data2['C']

# Determine number of timesteps
num_timesteps = C1.shape[0] // 100

# Function to get the 100x100 grid for a given timestep
def get_grid(C, timestep):
    return C[timestep * 100:(timestep + 1) * 100, :]

# Function to annotate the heatmap
def annotate_heatmap(im, data, ax, fmt="{:.1f}", text_colors=["black", "white"], threshold=None, skip=5):
    if threshold is None:
        threshold = (np.max(data) + np.min(data)) / 2.

    texts = []
    for i in range(0, data.shape[0], skip):
        for j in range(0, data.shape[1], skip):
            val = data[i, j]
            text_color = text_colors[int(val > threshold)]
            text = ax.text(j, i, fmt.format(val), ha="center", va="center", color=text_color, fontsize=5)
            texts.append(text)
    return texts

# Create figure and axes
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Set global color range for consistency
vmin = min(np.min(C1), np.min(C2))
vmax = max(np.max(C1), np.max(C2))

# Initial grid
grid1 = get_grid(C1, 0)
grid2 = get_grid(C2, 0)

# Initial heatmaps
im1 = ax[0].imshow(grid1, vmin=vmin, vmax=vmax, cmap='plasma', origin='lower')
ax[0].set_title("Height 2 - Timestep 0")
ax[0].set_xlabel("X Grid")
ax[0].set_ylabel("Y Grid")

im2 = ax[1].imshow(grid2, vmin=vmin, vmax=vmax, cmap='plasma', origin='lower')
ax[1].set_title("Height 5 - Timestep 0")
ax[1].set_xlabel("X Grid")
ax[1].set_ylabel("Y Grid")

# Add colorbars
cbar1 = fig.colorbar(im1, ax=ax[0])
cbar1.set_label('Concentration')

cbar2 = fig.colorbar(im2, ax=ax[1])
cbar2.set_label('Concentration')

# Store text handles
text1 = []
text2 = []

# Update function for animation
def update(frame):
    global text1, text2

    for txt in text1 + text2:
        txt.remove()

    grid1 = get_grid(C1, frame)
    grid2 = get_grid(C2, frame)

    im1.set_data(grid1)
    im2.set_data(grid2)

    ax[0].set_title(f"Height 2 - Timestep {frame}")
    ax[1].set_title(f"Height 5 - Timestep {frame}")

    text1 = annotate_heatmap(im1, grid1, ax[0], skip=5)
    text2 = annotate_heatmap(im2, grid2, ax[1], skip=5)

    return [im1, im2] + text1 + text2

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_timesteps, interval=100, blit=False)

# Save or show
action = "save"  # change to "show" if you want to display instead

if action == "show":
    plt.show()
elif action == "save":
    video_filename = f"concentration_grid_annotated_{timestamp}.mp4"
    print(f"Saving animation to {video_filename}...")
    writervideo = animation.FFMpegWriter(fps=30)
    ani.save(video_filename, writer=writervideo, dpi=200)
    print(f"Saved as {video_filename}")
