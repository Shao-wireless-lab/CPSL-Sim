import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Use GUI backend
matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 16})

# -------------------- Constants --------------------
target_x, target_y = 80, 60
labels = ['No Meander', 'Small Meander', 'Medium Meander']
colors = ['green', 'blue', 'red']  # For CDF plot
flux_color_plot1 = 'red'  # Use only red for Fluxotaxis in Plot 1
marl_color_plot1 = 'blue'  # Use only blue for MARL in Plot 1
example_indices = [0, 100, 200]

json_paths = [
    "/home/ece213/CPSL-Sim/results/test_results/fluxotaxis/test_easy_80_60_2025-04-11-15-08/episodes_traj/trajectory_1241_1321560341.json",
    "/home/ece213/CPSL-Sim/results/test_results/fluxotaxis/test_medium_80_60_2025-04-11-15-28/episodes_traj/trajectory_1426_3964150743.json",
    "/home/ece213/CPSL-Sim/results/test_results/fluxotaxis/test_hard_80_60_2025-04-11-15-48/episodes_traj/trajectory_1642_3341129303.json"
]

case_folders = [
    "/home/ece213/CPSL-Sim/results/test_results/fluxotaxis/test_easy_80_60_2025-04-11-15-08/episodes_traj",
    "/home/ece213/CPSL-Sim/results/test_results/fluxotaxis/test_medium_80_60_2025-04-11-15-28/episodes_traj",
    "/home/ece213/CPSL-Sim/results/test_results/fluxotaxis/test_hard_80_60_2025-04-11-15-48/episodes_traj"
]

# -------------------- Load CSV --------------------
csv_file_path = "fluxotaxis_all_extended.csv"
df = pd.read_csv(csv_file_path, header=None)
num_trials = 300
x_positions = np.array([df.iloc[:, 1 + 2*i].values for i in range(num_trials)])
y_positions = np.array([df.iloc[:, 2 + 2*i].values for i in range(num_trials)])
time = df.iloc[:, 0].values
time_mask = time > 140
time_trimmed = time[time_mask]

# -------------------- First Figure: Trajectories + Error vs Time --------------------
fig1, axs1 = plt.subplots(
    2, 3,
    figsize=(21, 10),
    gridspec_kw={'hspace': 0.5},  # adjust this value to increase space
    constrained_layout=False
)

subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
centroid_trajs = []

# ----------- Trajectories (Top Row) -----------
for i, (json_path, label, _, idx) in enumerate(zip(json_paths, labels, colors, example_indices)):
    with open(json_path, "r") as f:
        data = json.load(f)

    centroid_traj = []
    for entry in data:
        if "infos" in entry:
            first_agent_info = next(iter(entry["infos"].values()), {})
            centroid = first_agent_info.get("centroid_loc")
            if centroid:
                centroid_traj.append(centroid)

    centroid_traj = np.array(centroid_traj)
    centroid_trajs.append((centroid_traj, label, marl_color_plot1, idx))

    ax = axs1[0, i]
    ax.plot(x_positions[idx], y_positions[idx], color=flux_color_plot1, linestyle='--', label="Fluxotaxis")
    ax.scatter(target_x, target_y, color='green', s=60, marker='o', label='Emitter')
    ax.scatter(df.iloc[0, 1], df.iloc[0, 2], color='magenta', s=60, marker='o', label='UAV Initial Position')
    ax.plot(centroid_traj[:, 0], centroid_traj[:, 1], color=marl_color_plot1, linewidth=2, label='MARL')

    ax.set_xlim(40, 160)
    ax.set_ylim(40, 80)
    ax.set_yticks(np.arange(40, 81, 20))  # Set y-ticks with 20 intervals

    ax.set_title(f"{subplot_labels[i]} {label}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if i == 0:
        marl_line, = ax.plot(centroid_traj[:, 0], centroid_traj[:, 1], color=marl_color_plot1, linewidth=2,
                             label='MARL')
        flux_line, = ax.plot(x_positions[idx], y_positions[idx], color=flux_color_plot1, linestyle='--', label="Fluxotaxis")
        emitter_scatter = ax.scatter(target_x, target_y, color='green', s=60, marker='o', label='Emitter')
        uav_scatter = ax.scatter(df.iloc[0, 1], df.iloc[0, 2], color='magenta', s=60, marker='o',
                                 label='UAV Initial Position')

        handles = [marl_line, flux_line, emitter_scatter, uav_scatter]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc='lower left', fontsize='small')

# ----------- Error vs Time (Bottom Row) -----------
for i, (centroid_traj, label, _, idx) in enumerate(centroid_trajs):
    ax = axs1[1, i]
    uav_error_trimmed = np.sqrt((x_positions[idx, time_mask] - target_x) ** 2 +
                                (y_positions[idx, time_mask] - target_y) ** 2)
    shifted_time = time_trimmed - time_trimmed[0]
    ax.plot(shifted_time, uav_error_trimmed, color=flux_color_plot1, linestyle='--', label='Fluxotaxis')

    centroid_error_trimmed = np.sqrt((centroid_traj[:, 0] - target_x) ** 2 +
                                     (centroid_traj[:, 1] - target_y) ** 2)
    centroid_time_aligned = 0.05 * np.arange(len(centroid_traj))
    ax.plot(centroid_time_aligned, centroid_error_trimmed, color=marl_color_plot1, linestyle='-', label='MARL')


    ax.set_title(f"{subplot_labels[3 + i]} {label}")

    if i == 0:
        marl_line, = ax.plot(centroid_time_aligned, centroid_error_trimmed, color=marl_color_plot1, linestyle='-',
                             label='MARL')
        flux_line, = ax.plot(shifted_time, uav_error_trimmed, color=flux_color_plot1, linestyle='--',
                             label='Fluxotaxis')

        handles = [marl_line, flux_line]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc='upper right', fontsize='small')

# ----------- Titles -----------
# Top global title
fig1.suptitle("UAV Trajectories under Wind Conditions", fontsize=20, y=0.955)

# Middle section title (slightly above bottom row)
fig1.text(0.5, 0.475, "Distance to Emitter Over Time", ha='center', va='center', fontsize=20)

# Save
fig1.savefig("/home/ece213/CPSL-Sim/results/plots/test_with_fluxotaxis_trajectory_error.png", dpi=300,
             bbox_inches='tight')

# -------------------- Combined Final Distance CDF Figure --------------------
fig3, ax3 = plt.subplots(figsize=(8, 6))  # Single CDF plot

location_error_to_target = np.sqrt((x_positions - target_x)**2 + (y_positions - target_y)**2)
location_error_trimmed = location_error_to_target[:, time_mask]
fluxotaxis_groups = [(0, 100), (100, 200), (200, 300)]
fluxotaxis_cdfs = [np.sort(location_error_trimmed[start:end].flatten()) for start, end in fluxotaxis_groups]
fluxotaxis_percentiles = [np.linspace(0, 1, len(cdf)) for cdf in fluxotaxis_cdfs]

all_marl_dists = []
for i, folder in enumerate(case_folders):
    dist_list = []
    for file in os.listdir(folder):
        if file.endswith(".json") and "trajectory" in file:
            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)

            last_info = None
            recording = False
            for entry in data:
                if "infos" in entry:
                    first_agent_info = next(iter(entry["infos"].values()), {})
                    if not recording and first_agent_info.get("detection", 0) == 1:
                        recording = True
                    if recording:
                        last_info = first_agent_info

            if last_info and "final_cent_dist_to_emitter" in last_info:
                dist_list.append(last_info["final_cent_dist_to_emitter"])

    dist_sorted = np.sort(dist_list)
    all_marl_dists.append(dist_sorted)

x_max = np.max(np.concatenate(all_marl_dists + fluxotaxis_cdfs))
for i in range(3):
    # MARL = solid
    ax3.plot(all_marl_dists[i], np.linspace(0, 1, len(all_marl_dists[i])),
             color=colors[i], linestyle='-')
    # Fluxotaxis = dashed
    ax3.plot(fluxotaxis_cdfs[i], fluxotaxis_percentiles[i],
             color=colors[i], linestyle='--')

# Custom Legend
legend_lines = [
    Line2D([0], [0], color='green', lw=3),
    Line2D([0], [0], color='blue', lw=3),
    Line2D([0], [0], color='red', lw=3),
    Line2D([0], [0], color='black', linestyle='-', lw=3),
    Line2D([0], [0], color='black', linestyle='--', lw=3)
]
legend_labels = [
    "No Meander",
    "Small Meander",
    "Medium Meander",
    "MARL",
    "Fluxotaxis"
]
ax3.legend(legend_lines, legend_labels, loc='lower right')

#ax3.set_title("Final Distance to Emitter CDF")
ax3.set_xlabel("Final Distance to Emitter (m)")
ax3.set_ylabel("Cumulative Probability")
ax3.set_xlim(0, x_max)
ax3.set_ylim(0, 1.01)
plt.tight_layout()
fig3.savefig("/home/ece213/CPSL-Sim/results/plots/test_with_fluxotaxis_cdf.png", dpi=300)

