import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from datetime import datetime

matplotlib.use('TkAgg')

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

# Load .mat files
mat_data1 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/plume_data/Plume-C-Data-Height-2_G=3.mat")
mat_data2 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/plume_data/Plume-C-Data-Height-5_G=3.mat")

C1 = mat_data1['C']
C2 = mat_data2['C']

num_timesteps = C1.shape[0] // 100

def get_grid(C, timestep):
    return C[timestep * 100:(timestep + 1) * 100, :]

def get_top_n_mask(data, top_n=50):
    mask = np.zeros_like(data, dtype=bool)
    flat = data.flatten()
    if top_n >= len(flat):
        top_indices = np.argsort(flat)[::-1]
    else:
        top_indices = np.argpartition(flat, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]

    for idx in top_indices:
        i, j = divmod(idx, data.shape[1])
        mask[i, j] = True
    return mask

# === Crop Settings ===
crop_enabled = True
x_range_meters = (60, 160)  # meters
y_range_meters = (20, 100)  # meters
cell_size = 2  # meters per grid cell

# Convert to cell indices
x_start_idx = x_range_meters[0] // cell_size
x_end_idx = x_range_meters[1] // cell_size
y_start_idx = y_range_meters[0] // cell_size
y_end_idx = y_range_meters[1] // cell_size

def create_annotated_frame(ax, data, top_mask, frame):
    ax.clear()

    # Transpose to match Cartesian layout
    data = data.T
    top_mask = top_mask.T

    # Crop if enabled
    if crop_enabled:
        data = data[y_start_idx:y_end_idx, x_start_idx:x_end_idx]
        top_mask = top_mask[y_start_idx:y_end_idx, x_start_idx:x_end_idx]

    height, width = data.shape
    x_min = x_range_meters[0]
    x_max = x_range_meters[1]
    y_min = y_range_meters[0]
    y_max = y_range_meters[1]

    # Set coordinate system
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max + 1, 10))  # Every 10 meters
    ax.set_yticks(np.arange(y_min, y_max + 1, 10))  # Every 10 meters
    ax.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    rgba = np.ones((height, width, 4))  # RGBA image
    rgba[:, :, :3] = 1.0  # white base
    rgba[top_mask, :] = [0.56, 0.93, 0.56, 1.0]  # light green for top-N

    # Adjust image display range to match real-world coordinates
    ax.imshow(rgba, origin='lower', aspect='equal',
              extent=[x_min, x_max, y_min, y_max])

    for i in range(height):
        for j in range(width):
            if top_mask[i, j]:
                x_coord = x_min + j * cell_size + cell_size / 2
                y_coord = y_min + i * cell_size + cell_size / 2
                ax.text(x_coord, y_coord, f"{data[i, j]:.2f}",
                        ha='center', va='center', color='black', fontsize=4.5)

    # Show timestep in top-left of the plot
    ax.text(x_min + 5, y_max - 5, f"Timestep {frame}",
            fontsize=10, color='black', ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # === Add Emitter Marker ===
    emitter_x, emitter_y = 80, 60
    ax.plot(emitter_x, emitter_y, marker='o', color='green', markersize=6)
    #ax.text(emitter_x + 2, emitter_y, "emitter", color='green', fontsize=8, va='center')
    ax.text(emitter_x - 2, emitter_y, "emitter", color='green', fontsize=8, va='center', ha='right')



def generate_video(C, height_label, save_name):
    fig, ax = plt.subplots(figsize=(8, 8))  # Smaller size to speed up rendering

    def update(frame):
        grid = get_grid(C, frame)
        top_mask = get_top_n_mask(grid, top_n=50)
        ax.set_title(f"{height_label} - Timestep {frame}")
        create_annotated_frame(ax, grid, top_mask, frame)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=num_timesteps, interval=100, blit=False)

    print(f"Saving video: {save_name}")
    writervideo = animation.FFMpegWriter(fps=20)
    ani.save(save_name, writer=writervideo, dpi=200)
    print(f"Saved as {save_name}")
    plt.close(fig)

# Save videos (uncomment if needed)
generate_video(C1, "Height 2", f"height2_top50_subarea__G=3_{timestamp}.mp4")
generate_video(C2, "Height 5", f"height5_top50_subarea__G=3_{timestamp}.mp4")
