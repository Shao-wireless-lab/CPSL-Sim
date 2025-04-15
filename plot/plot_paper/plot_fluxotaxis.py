import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Use headless backend for saving plots
matplotlib.use('Agg')

# -------------------- Paths --------------------
csv_file_path = "/home/ece213/CPSL-Sim_2/plot/plot_paper/fluxotaxis_all_extended.csv"
json_paths = [
    "/home/ece213/CPSL-Sim_2/results/test_results/5/test_easy_80_60_2025-04-11-04-35/episodes_traj/trajectory_3200_1957495315.json",
    "/home/ece213/CPSL-Sim_2/results/test_results/5/test_medium_80_60_2025-04-11-04-24/episodes_traj/trajectory_3200_849202457.json",
    "/home/ece213/CPSL-Sim_2/results/test_results/5/test_hard_80_60_2025-04-11-04-14/episodes_traj/trajectory_3200_2369581521.json"
]

# -------------------- Constants --------------------
target_x, target_y = 80, 60
labels = ['No Meander', 'Small Meander', 'Medium Meander']
colors = ['black', 'blue', 'red']
example_indices = [0, 100, 200]

# -------------------- Load CSV --------------------
df = pd.read_csv(csv_file_path, header=None)
num_trials = 300
x_positions = np.array([df.iloc[:, 1 + 2*i].values for i in range(num_trials)])
y_positions = np.array([df.iloc[:, 2 + 2*i].values for i in range(num_trials)])
time = df.iloc[:, 0].values
time_mask = time > 140
time_trimmed = time[time_mask]

# -------------------- Storage --------------------
centroid_trajs = []

# -------------------- Figure 1: Trajectory Plots --------------------
fig1, axs1 = plt.subplots(1, 3, figsize=(21, 6))

for i, (label, color, idx, json_path) in enumerate(zip(labels, colors, example_indices, json_paths)):
    with open(json_path, "r") as f:
        data = json.load(f)

    centroid_traj = []
    emitter_loc = None

    for entry in data:
        if "infos" in entry:
            first_agent_info = next(iter(entry["infos"].values()), {})
            if emitter_loc is None and "target_loc" in first_agent_info:
                emitter_loc = first_agent_info["target_loc"]
            centroid = first_agent_info.get("centroid_loc")
            if centroid:
                centroid_traj.append(centroid)

    centroid_traj = np.array(centroid_traj) if centroid_traj else None
    centroid_trajs.append((centroid_traj, label, color, idx))

    axs1[i].plot(x_positions[idx], y_positions[idx], color='purple', label="Fluxotaxis method")
    axs1[i].scatter(target_x, target_y, color='green', s=60, marker='o', label='Emitter')
    axs1[i].scatter(df.iloc[0, 1], df.iloc[0, 2], color='magenta', s=60, marker='o', label='UAV Initial Position')
    if centroid_traj is not None:
        axs1[i].plot(centroid_traj[:, 0], centroid_traj[:, 1], color=color, linewidth=2, label='DRL method')

    axs1[i].set_xlim(40, 160)
    axs1[i].set_ylim(40, 100)
    axs1[i].set_title(f"Trajectory - {label}")
    axs1[i].set_xlabel("X (m)")
    axs1[i].set_ylabel("Y (m)")
    axs1[i].legend()

plt.tight_layout()
fig1.savefig("/home/ece213/CPSL-Sim_2/results/plots/trajectory.png", dpi=300)

# -------------------- Figure 2: Error vs Time --------------------
fig2, axs2 = plt.subplots(1, 3, figsize=(21, 6))

for i, (centroid_traj, label, color, idx) in enumerate(centroid_trajs):
    uav_error_trimmed = np.sqrt((x_positions[idx, time_mask] - target_x)**2 +
                                (y_positions[idx, time_mask] - target_y)**2)
    shifted_time = time_trimmed - time_trimmed[0]
    axs2[i].plot(shifted_time, uav_error_trimmed, color='purple', linestyle='--', label='Fluxotaxis method')

    if centroid_traj is not None:
        centroid_error_trimmed = np.sqrt((centroid_traj[:, 0] - target_x)**2 +
                                         (centroid_traj[:, 1] - target_y)**2)
        centroid_time_aligned = 0.05 * np.arange(len(centroid_traj))
        axs2[i].plot(centroid_time_aligned, centroid_error_trimmed, color=color, linestyle='-', label='DRL method')

    axs2[i].set_title(f"Location Error vs Time - {label}")
    axs2[i].set_xlabel("Time Since Activation (s)")
    axs2[i].set_ylabel("Error to Target (m)")
    axs2[i].legend()

plt.tight_layout()
fig2.savefig("/home/ece213/CPSL-Sim_2/results/plots/distance_error.png", dpi=300)

# -------------------- Figure 3: CDF of Location Error --------------------
fig3, ax3 = plt.subplots(figsize=(8, 6))

location_error_to_target = np.sqrt((x_positions - target_x)**2 + (y_positions - target_y)**2)
location_error_trimmed = location_error_to_target[:, time_mask]
groups = [(0, 100), (100, 200), (200, 300)]

for i, (start, end) in enumerate(groups):
    all_errors = location_error_trimmed[start:end].flatten()
    sorted_errors = np.sort(all_errors)
    cdf = np.linspace(0, 1, len(sorted_errors))
    ax3.plot(sorted_errors, cdf, color=colors[i], label=labels[i])

ax3.set_xlabel("Location Error (m)")
ax3.set_ylabel("Cumulative Probability")
ax3.set_title("CDF of Location Error After 140s")
ax3.set_xlim(left=0)
ax3.set_ylim(0, 1.01)
ax3.legend()

plt.tight_layout()
fig3.savefig("/home/ece213/CPSL-Sim_2/results/plots/error_cdf.png", dpi=300)
