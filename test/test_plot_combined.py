import json
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib
from scipy.ndimage import zoom

from datetime import datetime

matplotlib.use('TkAgg')

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")


# Load trajectory JSON data
with open("/home/ece213/CPSL-Sim/env_trajectory_2025-03-03-13-29_1227.json", "r") as f:
    data = json.load(f)

# Extract agent position data
agent_positions = {}
num_timesteps = len(data)

for timestep, timestep_data in enumerate(data):
    if "obs" in timestep_data:
        for agent_id, agent_data in timestep_data["obs"].items():
            if agent_id not in agent_positions:
                agent_positions[agent_id] = {"x": [], "y": []}
            raw_obs = agent_data.get("raw_obs", [])
            if len(raw_obs) >= 2:
                agent_positions[agent_id]["x"].append(raw_obs[0])
                agent_positions[agent_id]["y"].append(raw_obs[1])

# Load plume concentration data (height 5)
mat_data2 = scipy.io.loadmat("/home/ece213/CPSL-Sim/cuas/plume_data/Plume-C-Data-Height-5.mat")
C2 = mat_data2['C'].reshape((6000, 100, 100))
C2 = np.transpose(C2, (0, 2, 1))

# Resize concentration data from 100x100 to 200x200
C2_resized = zoom(C2, (1, 2, 2), order=1)

# Extract the last 3500 time steps
C2_processed = C2_resized[-3500:, :, :]

# Define threshold and create masked array
threshold = 0
masked_C2 = np.ma.masked_less_equal(C2_processed, threshold)

# Setup custom colormap
colors = [(1, 1, 1, 0), (0, 0.5, 0, 1)]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("white_to_green", colors, N=256)

# Set up figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Agent Trajectories and Concentration (Height=5)")
ax.grid()

# Initialize concentration plot
conc_plot = ax.imshow(masked_C2[0], cmap=custom_cmap, interpolation='nearest', origin='lower', extent=[0,200,0,200])
plt.colorbar(conc_plot, ax=ax)

# Initialize plots for agent trajectories
lines, scatters = {}, {}
for agent_id in agent_positions.keys():
    line, = ax.plot([], [], '-', label=f"Agent {agent_id}")
    scatter = ax.scatter([], [], edgecolors='black', s=50)
    lines[agent_id] = line
    scatters[agent_id] = scatter

# Add emitter position (scaled to 2x)
emitter_position = [40 * 2, 30 * 2]  # Scaling emitter position
ax.plot(emitter_position[0], emitter_position[1], 'ro', markersize=5, label="Emitter")
ax.text(emitter_position[0], emitter_position[1] - 3, "Emitter", fontsize=5, color='red',
        verticalalignment='top', horizontalalignment='center')

ax.legend()

# Add time step text
time_text = ax.text(5, 190, "Time step: 0", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.7))

# Update function for animation
def update(frame):
    if frame < len(masked_C2):
        conc_plot.set_data(masked_C2[frame])

    for agent_id in agent_positions.keys():
        if frame < len(agent_positions[agent_id]["x"]):
            x_traj = agent_positions[agent_id]["x"][:frame+1]
            y_traj = agent_positions[agent_id]["y"][:frame+1]

            lines[agent_id].set_data(x_traj, y_traj)
            scatters[agent_id].set_offsets(np.array([[x_traj[-1], y_traj[-1]]]))

    time_text.set_text(f"Time step: {frame}")

    # Add "Emitter Found!" at the last frame
    if frame == num_timesteps - 1:
        ax.text(100, 180, "Emitter Found!", fontsize=14, color='blue', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))

    return [conc_plot, time_text] + list(lines.values()) + list(scatters.values())

# Option to choose frame skip
use_frame_skip = False
frame_skip = 10

frames = range(0, num_timesteps, frame_skip) if use_frame_skip else range(num_timesteps)

ani = animation.FuncAnimation(
    fig, update, frames=frames, interval=100, blit=False
)

# Choose action: "show" or "save"
action = "save"  # Change to "show" if you prefer to display the animation directly

if action == "show":
    plt.show()
elif action == "save":
    video_filename = f"demo_{timestamp}.mp4"
    print(f"Animation is starting to save as {video_filename}")
    writervideo = animation.FFMpegWriter(fps=30)
    ani.save(video_filename, writer=writervideo, dpi=200)
    print(f"Animation saved as {video_filename}")
