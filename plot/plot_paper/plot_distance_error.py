import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
matplotlib.use('TkAgg')

# === Base result folder ===
base_dir = "/home/ece213/CPSL-Sim_2/results/test_results/"

# === Folder: label mappings ===
folder_map = {
    "test_easy_80_60_2025-04-11-09-23":      ("No Meander", "Emitter [80, 60]"),
    "test_medium_80_60_2025-04-11-10-42":   ("Small Meander", "Emitter [80, 60]"),
    "test_hard_80_60_2025-04-11-14-16":  ("Medium Meander", "Emitter [80, 60]"),
    "test_easy_60_120_2025-04-11-10-11":     ("No Meander", "Emitter [60, 120]"),
    "test_medium_60_120_2025-04-11-10-25":  ("Small Meander", "Emitter [60, 120]"),
    "test_hard_60_120_2025-04-11-10-26": ("Medium Meander", "Emitter [60, 120]"),
}

# === Load tolerance from config ===
sim_config_path = "/home/ece213/CPSL-Sim_2/configs/sim_config.cfg"
with open(sim_config_path, "r") as f:
    sim_config = json.load(f)
tolerance = sim_config['env_config']["tolerance"]

# === Collect data ===
plot_data = []

for folder, (meander_label, emitter_loc) in folder_map.items():
    folder_path = os.path.join(base_dir, folder, "episodes_traj")
    if not os.path.exists(folder_path):
        print(f"⚠️ Missing folder: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".json") and "trajectory" in file:
            json_path = os.path.join(folder_path, file)
            with open(json_path, "r") as f:
                data = json.load(f)

            last_info = None
            recording = False
            for entry in data:
                if "infos" in entry:
                    first_agent_info = next(iter(entry["infos"].values()), {})
                    detection = first_agent_info.get("detection", 0)
                    if not recording and detection == 1:
                        recording = True
                    if recording:
                        last_info = first_agent_info

            if last_info and "final_cent_dist_to_emitter" in last_info:
                plot_data.append({
                    "distance": last_info["final_cent_dist_to_emitter"],
                    "offset": last_info.get("final_cent_dist_to_emitter_offset", None),
                    "meander": meander_label,
                    "location": emitter_loc
                })

# === Grouped data ===
meander_levels = ['No Meander', 'Small Meander', 'Medium Meander']
emitter_locations = ['Emitter [80, 60]', 'Emitter [60, 120]']
color_map = {'No Meander': 'black', 'Small Meander': 'blue', 'Medium Meander': 'red'}

group_count = len(meander_levels)
bar_width = 0.35
x = np.arange(group_count)

raw_means = []
offset_means = []

for m in meander_levels:
    for loc in emitter_locations:
        entries = [d for d in plot_data if d["meander"] == m and d["location"] == loc]
        d_vals = [e["distance"] for e in entries if e["distance"] is not None]
        o_vals = [e["offset"] for e in entries if e["offset"] is not None]

        raw_means.append(np.mean(d_vals) if d_vals else 0)
        offset_means.append(np.mean(o_vals) if o_vals else 0)

# === Success rate data ===
scenarios_80_60 = [0.99, 0.89, 0.83]
scenarios_60_120 = [1.00, 0.93, 0.76]
scenarios_80_60_pct = [r * 100 for r in scenarios_80_60]
scenarios_60_120_pct = [r * 100 for r in scenarios_60_120]

# === Plotting all three subplots ===
fig, axs = plt.subplots(1, 3, figsize=(21, 6), sharey=False)

# === Subplot 1: Success rate ===
x_labels = meander_levels
x_pos = np.arange(len(x_labels))

for i, val in enumerate(scenarios_80_60_pct):
    axs[0].bar(x_pos[i] - bar_width / 2, val, width=bar_width,
               color=color_map[meander_levels[i]], edgecolor='black')
for i, val in enumerate(scenarios_60_120_pct):
    axs[0].bar(x_pos[i] + bar_width / 2, val, width=bar_width,
               color=color_map[meander_levels[i]], edgecolor='white', hatch='//')

# Annotate bars
for i, val in enumerate(scenarios_80_60_pct):
    axs[0].text(x_pos[i] - bar_width / 2, val + 1, f"{val:.0f}%", ha='center', fontsize=9)
for i, val in enumerate(scenarios_60_120_pct):
    axs[0].text(x_pos[i] + bar_width / 2, val + 1, f"{val:.0f}%", ha='center', fontsize=9)

axs[0].set_ylim(0, 120)
axs[0].set_ylabel("Success Rate (%)")
axs[0].set_title("Success Rate by Wind Condition")
axs[0].set_xticks(x_pos)
axs[0].set_xticklabels(x_labels)
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# Hide the top y-tick (120)
yticks = axs[0].get_yticks()
axs[0].set_yticks([yt for yt in yticks if yt < 120])

# === Subplot 2: Final distance ===
means_em1 = raw_means[::2]
means_em2 = raw_means[1::2]

for i in range(len(meander_levels)):
    axs[1].bar(x[i] - bar_width / 2, means_em1[i], width=bar_width,
               color=color_map[meander_levels[i]], edgecolor='black')
    axs[1].bar(x[i] + bar_width / 2, means_em2[i], width=bar_width,
               color=color_map[meander_levels[i]], edgecolor='white', hatch='//')

axs[1].set_title("Final Distance to Emitter")
axs[1].set_xticks(x)
axs[1].set_xticklabels(meander_levels)
axs[1].set_ylabel("Distance (m)")
axs[1].grid(True, axis='y')
axs[1].axhline(y=tolerance, color='green', linestyle='--', linewidth=2)
axs[1].text(-0.4, tolerance + 0.2, f'Tolerance: {tolerance:.2f}m',
            color='green', fontsize=10, va='bottom')

# === Subplot 3: Offset distance ===
means_em1 = offset_means[::2]
means_em2 = offset_means[1::2]

for i in range(len(meander_levels)):
    axs[2].bar(x[i] - bar_width / 2, means_em1[i], width=bar_width,
               color=color_map[meander_levels[i]], edgecolor='black')
    axs[2].bar(x[i] + bar_width / 2, means_em2[i], width=bar_width,
               color=color_map[meander_levels[i]], edgecolor='white', hatch='//')

axs[2].set_title("Offset Distance to Emitter")
axs[2].set_xticks(x)
axs[2].set_xticklabels(meander_levels)
axs[2].set_ylabel("Distance (m)")
axs[2].grid(True, axis='y')
axs[2].axhline(y=tolerance, color='green', linestyle='--', linewidth=2)
axs[2].text(-0.4, tolerance + 0.2, f'Tolerance: {tolerance:.2f}m',
            color='green', fontsize=10, va='bottom')

# === Combined Legend ===
legend_elements = [
    Patch(facecolor='black', label='No Meander'),
    Patch(facecolor='blue', label='Small Meander'),
    Patch(facecolor='red', label='Medium Meander'),
    Patch(facecolor='gray', edgecolor='black', label='Emitter [80, 60]'),
    Patch(facecolor='gray', edgecolor='white', hatch='//', label='Emitter [60, 120]')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=5, fontsize=10)

# === Save and show ===
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = "/home/ece213/CPSL-Sim_2/results/plots/success_dist.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"Plot saved to: {save_path}")
plt.show()
plt.close()
