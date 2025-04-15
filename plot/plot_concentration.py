import os
import time
import json
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from datetime import datetime

matplotlib.use('TkAgg')

# === Load the plume_scenarios.json and sim_config.cfg (actually JSON) for plume scenarios ===
scenario_json_path = "/home/ece213/CPSL-Sim_2/configs/plume_scenarios.json"

# Load plume scenarios config
with open(scenario_json_path, "r") as f:
    plume_scenarios = json.load(f)

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

def create_annotated_frame(ax, data, top_mask, frame, emitter_x, emitter_y):
    ax.clear()
    data = data.T
    top_mask = top_mask.T
    height, width = data.shape
    cell_size = 2
    env_width = width * cell_size
    env_height = height * cell_size

    ax.set_xlim(0, env_width)
    ax.set_ylim(0, env_height)
    ax.set_xticks(np.arange(0, env_width + 1, 40))
    ax.set_yticks(np.arange(0, env_height + 1, 40))
    ax.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    rgba = np.ones((height, width, 4))
    rgba[:, :, :3] = 1.0
    rgba[top_mask, :] = [0.56, 0.93, 0.56, 1.0]

    ax.imshow(rgba, origin='lower', aspect='equal', extent=[0, env_width, 0, env_height])

    for i in range(height):
        for j in range(width):
            if top_mask[i, j]:
                x_coord = j * cell_size + cell_size / 2
                y_coord = i * cell_size + cell_size / 2
                ax.text(x_coord, y_coord, f"{data[i, j]:.2f}",
                        ha='center', va='center', color='black', fontsize=6)

    # Add emitter marker from scenario
    ax.plot(emitter_x, emitter_y, marker='o', color='darkgreen', markersize=6)
    ax.text(emitter_x - 2, emitter_y, "emitter", color='darkgreen', fontsize=8, va='center', ha='right')

    ax.text(5, env_height - 5, f"Timestep {frame}",
            fontsize=10, color='black', ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

def generate_video(C, emitter_x, emitter_y, save_name, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    num_timesteps = C.shape[0] // 100

    def update(frame):
        grid = get_grid(C, frame)
        top_mask = get_top_n_mask(grid, top_n=50)
        ax.set_title(f"{title} - Timestep {frame}")
        create_annotated_frame(ax, grid, top_mask, frame, emitter_x, emitter_y)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=num_timesteps, interval=100, blit=False)
    print(f"Saving video: {save_name}")
    writervideo = animation.FFMpegWriter(fps=30)

    start_time = time.time()
    ani.save(save_name, writer=writervideo, dpi=200)
    end_time = time.time()

    duration = end_time - start_time
    print(f"✅ Saved as {save_name} | Time taken: {duration:.2f} seconds")
    plt.close(fig)

# Loop through scenarios in JSON
for name, config in plume_scenarios.items():
    data_dir = config["data_dir"]
    emitter_x = config["target"]["x"]
    emitter_y = config["target"]["y"]
    mat_file = os.path.join(data_dir, "Plume-C-Data-Height-5.mat")

    if not os.path.isfile(mat_file):
        print(f"❌ File not found: {mat_file}")
        continue

    print(f"📂 Processing {name} from {mat_file}")
    mat_data = scipy.io.loadmat(mat_file)
    if 'C' not in mat_data:
        print(f"❌ No 'C' matrix found in {mat_file}")
        continue

    C = mat_data['C']
    save_name = os.path.join(data_dir, f"Plume-C-{name}-Height-5.mp4")
    generate_video(C, emitter_x, emitter_y, save_name, title=name)
