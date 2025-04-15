import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

# Load the trajectory JSON
with open("/home/ece213/CPSL-Sim_2/results/test_results/test_2025-03-30-19-28/trajectory_636_2452459656.json", "r") as f:

#with open("/home/ece213/CPSL-Sim_2/results/test_results/test_2025-03-28-13-29/trajectory_764.json", "r") as f:

    data = json.load(f)

centroid_traj = []
anchor_traj = []
valid_timesteps = []  # to track original timestep indices
recording = False

# Iterate over timesteps
for timestep, entry in enumerate(data):
    if "infos" in entry:
        first_agent_info = next(iter(entry['infos'].values()), {})

        no_detection = first_agent_info.get("no_detection", 1)  # assume 1 (invalid) if missing

        if not recording and no_detection == 0:
            recording = True  # start recording from here

        if recording:
            centroid_loc = first_agent_info.get("centroid_loc")
            anchor_loc = first_agent_info.get("anchor_loc")

            if centroid_loc is not None and anchor_loc is not None:
                centroid_traj.append(centroid_loc)
                anchor_traj.append(anchor_loc)
                valid_timesteps.append(timestep)

# Convert to arrays
centroid_array = np.array(centroid_traj)
anchor_array = np.array(anchor_traj)
timesteps_array = np.array(valid_timesteps)

# Distance deltas
centroid_diff = np.linalg.norm(np.diff(centroid_array, axis=0), axis=1)
centroid_diff = np.insert(centroid_diff, 0, 0)

anchor_diff = np.linalg.norm(np.diff(anchor_array, axis=0), axis=1)
anchor_diff = np.insert(anchor_diff, 0, 0)

# Distance between centroid and anchor
anchor_centroid_dist = np.linalg.norm(centroid_array - anchor_array, axis=1)

# Plot 1: Trajectories with selective annotations
plt.figure(figsize=(8, 6))
plt.plot(centroid_array[:, 0], centroid_array[:, 1], color='blue',marker='o', markersize=2, linewidth=1, label="Centroid Trajectory")
plt.plot(anchor_array[:, 0], anchor_array[:, 1], color='red', marker='x', markersize=2, linewidth=1, linestyle='--', label="Anchor Trajectory")

# Annotate centroid every 50 steps
for i, t in enumerate(timesteps_array):
    if i % 50 == 0:
        plt.text(centroid_array[i, 0], centroid_array[i, 1], f"{t}", fontsize=10, color='blue', ha='right', va='bottom')

# Annotate anchor only when it changes
prev_anchor = None
for i, (anchor_pos, t) in enumerate(zip(anchor_array, timesteps_array)):
    if prev_anchor is None or not np.allclose(anchor_pos, prev_anchor):
        plt.text(anchor_pos[0], anchor_pos[1], f"A@{t}", fontsize=10, color='red', ha='left', va='top')
        prev_anchor = anchor_pos

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Centroid & Anchor Location Trajectories (Labeled)")
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.grid(True)
plt.legend()
plt.tight_layout()


# Plot 2: ΔDistance over time (using valid timesteps)
plt.figure(figsize=(10, 4))
plt.plot(timesteps_array, centroid_diff, color='orange', label="Centroid Distance Δ")
plt.plot(timesteps_array, anchor_diff, color='blue', label="Anchor Distance Δ")
plt.xlabel("Original Timestep")
plt.ylabel("Distance Δ")
plt.title("Centroid & Anchor Movement Distance Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot 3: Distance between anchor and centroid
plt.figure(figsize=(10, 4))
plt.plot(timesteps_array, anchor_centroid_dist, color='green', label="Centroid ↔ Anchor Distance")
plt.xlabel("Original Timestep")
plt.ylabel("Distance")
plt.title("Distance Between Centroid and Anchor Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()