import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import os

matplotlib.use('TkAgg')  # For matplotlib GUI

json_path = "/home/ece213/CPSL-Sim_2/results/test_results/test_2025-04-04-12-28/trajectory_725_1668442387.json"

# === Load JSON Data ===
with open(json_path, "r") as f:
    data = json.load(f)

centroid_traj = []
anchor_traj = []
valid_timesteps = []
emitter_loc = []

recording = False

# === Extract Data ===
for entry in data:
    if "infos" in entry:
        first_agent_info = next(iter(entry["infos"].values()), {})
        if "target_loc" in first_agent_info:
            emitter_loc = first_agent_info["target_loc"]
            break  # Stop after first valid emitter_loc



for timestep, entry in enumerate(data):
    if "infos" in entry:
        first_agent_info = next(iter(entry["infos"].values()), {})
        detection = first_agent_info.get("detection", 0)

        if not recording and detection == 1:
            recording = True

        if recording:
            centroid_loc = first_agent_info.get("centroid_loc")
            anchor_loc = first_agent_info.get("anchor_loc")

            if centroid_loc is not None and anchor_loc is not None:
                centroid_traj.append(centroid_loc)
                anchor_traj.append(anchor_loc)
                valid_timesteps.append(timestep)

# Convert to numpy arrays
centroid_array = np.array(centroid_traj)
anchor_array = np.array(anchor_traj)
timesteps_array = np.array(valid_timesteps)

# === Compute Distance Changes ===
centroid_diff = np.linalg.norm(np.diff(centroid_array, axis=0), axis=1)
centroid_diff = np.insert(centroid_diff, 0, 0)

anchor_diff = np.linalg.norm(np.diff(anchor_array, axis=0), axis=1)
anchor_diff = np.insert(anchor_diff, 0, 0)

anchor_centroid_dist = np.linalg.norm(centroid_array - anchor_array, axis=1)

# === Optional: Matplotlib Plots ===
plt.figure(figsize=(8, 6))
plt.plot(centroid_array[:, 0], centroid_array[:, 1], marker='o', markersize=2, linewidth=1, label="Centroid Trajectory")
plt.plot(anchor_array[:, 0], anchor_array[:, 1], marker='x', markersize=2, linewidth=1, linestyle='--', label="Anchor Trajectory")
for i, t in enumerate(timesteps_array):
    if i % 50 == 0:
        plt.text(centroid_array[i, 0], centroid_array[i, 1], f"{t}", fontsize=8, color='black', ha='right', va='bottom')
prev_anchor = None
for i, (anchor_pos, t) in enumerate(zip(anchor_array, timesteps_array)):
    if prev_anchor is None or not np.allclose(anchor_pos, prev_anchor):
        plt.text(anchor_pos[0], anchor_pos[1], f"A@{t}", fontsize=8, color='blue', ha='left', va='top')
        prev_anchor = anchor_pos


# === Plot emitter point ===
if emitter_loc is not None:
    plt.plot(emitter_loc[0], emitter_loc[1], 'ro', markersize=6, label="Emitter")
    plt.text(emitter_loc[0], emitter_loc[1], "Emitter", fontsize=10, color='green', ha='left', va='bottom')



plt.xlabel("X")
plt.ylabel("Y")
plt.title("Centroid & Anchor Location Trajectories")
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.grid(True)
plt.legend()
plt.tight_layout()

# === Plotly Interactive HTML Plot ===
fig = go.Figure()

# Centroid Trace
fig.add_trace(go.Scatter(
    x=centroid_array[:, 0],
    y=centroid_array[:, 1],
    mode='lines+markers+text',
    name='Centroid Trajectory',
    marker=dict(size=4),
    line=dict(width=1),
    text=[str(t) if i % 50 == 0 else '' for i, t in enumerate(timesteps_array)],
    textposition="top right",
    textfont=dict(size=10)
))

# Anchor Trace
fig.add_trace(go.Scatter(
    x=anchor_array[:, 0],
    y=anchor_array[:, 1],
    mode='lines+markers+text',
    name='Anchor Trajectory',
    marker=dict(symbol='x', size=5),
    line=dict(dash='dash', width=1),
    text=[
        f"A@{t}" if (i == 0 or not np.allclose(anchor_array[i], anchor_array[i - 1])) else ''
        for i, t in enumerate(timesteps_array)
    ],
    textposition="bottom left",
    textfont=dict(size=10, color='blue')
))

# === Add emitter to Plotly plot ===
if emitter_loc is not None:
    fig.add_trace(go.Scatter(
        x=[emitter_loc[0]],
        y=[emitter_loc[1]],
        mode='markers+text',
        name='Emitter',
        marker=dict(size=10, color='green'),
        text=["Emitter"],
        textposition="top right",
        textfont=dict(size=12, color='green')
    ))


fig.update_layout(
    title="Centroid & Anchor Trajectories (Interactive)",
    xaxis_title="X",
    yaxis_title="Y",
    width=800,
    height=600,
    showlegend=True
)


#output_dir = "plot_results"
# Save HTML into a subfolder called "plot_results" inside the original JSON folder
output_dir = os.path.join(os.path.dirname(json_path), "plot_results")
os.makedirs(output_dir, exist_ok=True)
#timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

json_filename = os.path.splitext(os.path.basename(json_path))[0]
save_path = os.path.join(output_dir, f"{json_filename}.html")

fig.write_html(save_path)
print(f"✅Interactive plot Saved at: {os.path.abspath(save_path)}")
