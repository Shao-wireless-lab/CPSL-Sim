# --- Unchanged imports ---
import json
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib
import os

from scipy.ndimage import zoom
from datetime import datetime
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

matplotlib.use('TkAgg')

# ----------------------- Input and Output Paths -----------------------
file_path = "/home/ece213/CPSL-Sim_2/results/test_results/fluxotaxis/test_easy_80_60_2025-04-11-15-08/episodes_traj/trajectory_1447_1451642945.json"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_save_path = f"/home/ece213/CPSL-Sim_2/results/plots"
os.makedirs(plot_save_path, exist_ok=True)
mat_data = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/plume_data/G=1_80_60/Plume-C-Data-Height-5.mat")

env_setup = r"Wind Condition: $\mathbf{No\ meander}$ | Emitter Location: [$\mathbf{80,60}$]"
tolerance_distance = 4  # meters

# -------------------- Helper Function --------------------
def find_all_detected(data):
    agents_detected = set()
    total_agents = None
    for timestep, entry in enumerate(data):
        if 'infos' in entry:
            if total_agents is None:
                total_agents = set(entry['infos'].keys())
            for agent_id, agent_data in entry['infos'].items():
                if 'detection' in agent_data and agent_data['detection'] == 1:
                    agents_detected.add(agent_id)
        if total_agents and agents_detected == total_agents:
            return timestep
    return len(data)

# -------------------- Load JSON and Plume Data --------------------
with open(file_path, 'r') as f:
    trajectory_data = json.load(f)

all_detected_time = find_all_detected(trajectory_data)

C_data = np.transpose(mat_data['C'].reshape((6000, 100, 100)), (0, 2, 1))
C_sliced = C_data[-3400:]

first_info = trajectory_data[0]["infos"]
any_agent_info = next(iter(first_info.values()))
shift = any_agent_info.get("shift", [0, 0])
target_loc = any_agent_info.get("target_loc", [0, 0])

C_shifted = np.array([np.roll(np.roll(C_sliced[t], shift[1], axis=0), shift[0], axis=1)
                      for t in range(C_sliced.shape[0])])
C_resized = zoom(C_shifted, (1, 2, 2), order=1)
binary_C = np.where(C_resized > 2.0, 1, 0)

# -------------------- Extract Agent and Obstacle Data --------------------
agent_positions = {}
obstacle_positions = {}
anchor_points = {}
num_timesteps = len(trajectory_data)

for timestep, timestep_data in enumerate(trajectory_data):
    if "obs" in timestep_data:
        for aid, adata in timestep_data["obs"].items():
            if aid not in agent_positions:
                agent_positions[aid] = {"x": [], "y": []}
            raw_obs = adata.get("raw_obs", [])
            if len(raw_obs) >= 2:
                agent_positions[aid]["x"].append(raw_obs[0])
                agent_positions[aid]["y"].append(raw_obs[1])
    if "obstacle_locs" in timestep_data:
        obstacle_positions[timestep] = {k: (v["x"], v["y"])
                                        for k, v in timestep_data["obstacle_locs"].items()}
    if "infos" in timestep_data:
        for aid, adata in timestep_data["infos"].items():
            if aid not in anchor_points:
                anchor_points[aid] = []
            anchor = adata.get("anchor_loc")
            if anchor:
                if not anchor_points[aid] or anchor_points[aid][-1][1] != anchor:
                    anchor_points[aid].append((timestep, anchor))

# -------------------- Plot Setup --------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_aspect('equal')
ax.grid()
ax.set_title(f"{env_setup}")
ax.set_xlabel("Global X Position (m)")
ax.set_ylabel("Global Y Position (m)")

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Emitter', markerfacecolor='green', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Agent (UAV)', markerfacecolor='blue', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Anchor Point', markerfacecolor='red', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Obstacle', markerfacecolor='black', alpha=0.7, markersize=8),
    Line2D([0], [0], marker='o', color='lightgrey', label='Observation Range', alpha=0.3, markersize=10),
    Line2D([0], [0], marker='s', color='lightgreen', label='Plume', markersize=8, linestyle='None')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

conc_plot = ax.imshow(binary_C[0], cmap=mcolors.ListedColormap(["white", "lightgreen"]),
                      interpolation='nearest', origin='lower', extent=[0, 200, 0, 200])

agent_color = "blue"
agent_bodies = {}
obs_circles = {}
for agent_id in agent_positions:
    circle = Circle((0, 0), radius=1, edgecolor='black', facecolor=agent_color)
    ax.add_patch(circle)
    agent_bodies[agent_id] = circle

    obs = Circle((0, 0), radius=20, color='lightgreen', alpha=0.3)
    ax.add_patch(obs)
    obs_circles[agent_id] = obs

obstacle_circles = [Circle((0, 0), radius=4, color='black', alpha=0.7, visible=False)
                    for _ in range(20)]
for circle in obstacle_circles:
    ax.add_patch(circle)

ax.plot(*target_loc, 'go', markersize=5)
ax.text(target_loc[0] - 2, target_loc[1], "Emitter", fontsize=10, color='green',
        verticalalignment='center', horizontalalignment='right')

# UI Text
time_text = fig.text(0.25, 0.02, "Time step: -- | Time: -- s", ha="center", fontsize=12, color="black")
status_text = fig.text(0.75, 0.02, "Status: --", ha="center", fontsize=14, color='blue', fontweight='bold')
emitter_found_text = fig.text(0.5, 0.7, "Emitter Found!", ha="center", fontsize=60,
                              color="darkgreen", fontweight="bold", visible=False)

# Anchor points with fading alpha
anchor_dots = [
    ax.plot([], [], 'ro', markersize=2, alpha=0.5)[0],
    ax.plot([], [], 'ro', markersize=2, alpha=0.75)[0],
    ax.plot([], [], 'ro', markersize=2, alpha=1.0)[0]
]

# Tolerance Circle and Label
tolerance_circle = Circle((0, 0), radius=tolerance_distance, edgecolor='blue',
                          facecolor='none', linewidth=2.5, linestyle='--', visible=False)
ax.add_patch(tolerance_circle)
tolerance_label = ax.text(0, 0, '', color='blue', fontsize=10, fontweight='bold', visible=False)

# -------------------- Frame Update Function --------------------
def update(frame):
    is_freeze_frame = frame >= num_timesteps
    frame = min(frame, num_timesteps - 1)

    info = trajectory_data[frame].get("infos", {})
    should_zoom = False
    if frame >= 400:
        for aid, data in info.items():
            if data.get("detection") == 1 and data.get("current_cent_dist_to_emitter", 999) < 25:
                should_zoom = True
                break

    xlim = (max(target_loc[0] - 40, 0), min(target_loc[0] + 40, 200)) if should_zoom else (0, 200)
    ylim = (max(target_loc[1] - 40, 0), min(target_loc[1] + 40, 200)) if should_zoom else (0, 200)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    scaling_factor = 200 / max(xrange, yrange)

    agent_radius = 1 / scaling_factor
    obstacle_radius = 4 / scaling_factor

    for aid in agent_bodies:
        agent_bodies[aid].set_radius(agent_radius)
    for circle in obstacle_circles:
        circle.set_radius(obstacle_radius)

    conc_plot.set_data(binary_C[frame] if frame < len(binary_C) else binary_C[-1])
    is_stabilized = frame >= num_timesteps - 100 or is_freeze_frame

    if frame < all_detected_time:
        status_text.set_text("Status:\nDetection Phase, No Gas Detected")
        for obs in obs_circles.values(): obs.set_color('lightgreen')
    elif not is_stabilized:
        status_text.set_text("Status:\nDetected, Tracing the Emitter")
        for obs in obs_circles.values(): obs.set_color('lightcoral')
    else:
        status_text.set_text("Status:\nStabilized, Declaring the Emitter")
        for obs in obs_circles.values(): obs.set_color('lightskyblue')

    emitter_found_text.set_visible(is_freeze_frame)

    for aid in agent_positions:
        if frame < len(agent_positions[aid]["x"]):
            x, y = agent_positions[aid]["x"][frame], agent_positions[aid]["y"][frame]
            agent_bodies[aid].center = (x, y)
            obs_circles[aid].center = (x, y)

    for circle in obstacle_circles:
        circle.set_visible(False)
    if frame in obstacle_positions:
        for i, pos in enumerate(obstacle_positions[frame].values()):
            if i < len(obstacle_circles):
                obstacle_circles[i].center = pos
                obstacle_circles[i].set_visible(True)

    if not is_stabilized and anchor_points:
        first_agent = next(iter(anchor_points))
        all_anchors = [loc for t, loc in anchor_points[first_agent] if t <= frame]
        all_anchors = list(dict.fromkeys(map(tuple, all_anchors)))
        recent_anchors = all_anchors[-3:]
        for i in range(3):
            if i < len(recent_anchors):
                x, y = recent_anchors[-3 + i] if len(recent_anchors) >= 3 else recent_anchors[i]
                anchor_dots[i].set_data(x, y)
                anchor_dots[i].set_visible(True)
            else:
                anchor_dots[i].set_visible(False)
    else:
        for dot in anchor_dots:
            dot.set_visible(False)

    if is_freeze_frame:
        final_info = trajectory_data[-1].get("infos", {})
        first_agent_id = next(iter(final_info))
        final_centroid = final_info[first_agent_id].get("centroid_loc", None)
        if final_centroid:
            cx, cy = final_centroid
            tolerance_circle.center = (cx, cy)
            tolerance_circle.set_visible(True)
            tolerance_label.set_position((cx + 6, cy + 6))
            tolerance_label.set_text(f"Tolerance: {tolerance_distance} m")
            tolerance_label.set_visible(True)
        else:
            tolerance_circle.set_visible(False)
            tolerance_label.set_visible(False)
    else:
        tolerance_circle.set_visible(False)
        tolerance_label.set_visible(False)

    if frame < all_detected_time:
        time_text.set_text("Time step: -- | Time: -- s")
    else:
        adjusted_step = frame - all_detected_time
        real_time = adjusted_step * 0.05
        time_text.set_text(f"Time step: {adjusted_step} | Time: {real_time:.2f} s")

    return [conc_plot] + list(agent_bodies.values()) + list(obs_circles.values()) + \
           list(obstacle_circles) + anchor_dots + [emitter_found_text, tolerance_circle, tolerance_label]

# -------------------- Animate and Save --------------------
frames = list(range(num_timesteps)) + [num_timesteps] * 60
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

action = "save"
if action == "show":
    plt.show()
elif action == "save":
    video_filename = os.path.join(plot_save_path, f"demo_{timestamp}.mp4")
    print(f"Saving animation to {video_filename}")
    writer = animation.FFMpegWriter(fps=20)
    ani.save(video_filename, writer=writer, dpi=300)
    print(f"Animation saved at: {video_filename}")
