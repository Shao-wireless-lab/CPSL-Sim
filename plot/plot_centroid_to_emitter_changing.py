import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

trajectory_file = "/home/ece213/CPSL-Sim_2/results/test_results/test_2025-04-08-15-56/episodes_traj/trajectory_2688_170479296.json"

with open(trajectory_file, "r") as f:
    data = json.load(f)

# 初始化变量
centroid_traj = []
emitter_dist = []
valid_timesteps = []

recording = False
emitter_loc = None

for timestep, entry in enumerate(data):
    if "infos" in entry:
        first_agent_info = next(iter(entry["infos"].values()), {})
        detection = first_agent_info.get("detection", 1)

        # Valid timestep record trigger
        if not recording and detection == 1:
            recording = True
            print(f"✅ Recording started at timestep {timestep}")

        if recording:
            centroid_loc = first_agent_info.get("centroid_loc")
            if emitter_loc is None:
                emitter_loc = first_agent_info.get("target_loc")


            #print(f"Timestep {timestep}:")
            #print(f"  detection = {detection}")
            #print(f"  centroid_loc = {centroid_loc}")
            #print(f"  emitter_loc = {emitter_loc}")

            if centroid_loc is not None and emitter_loc is not None:
                centroid_traj.append(centroid_loc)
                dist = np.linalg.norm(np.array(centroid_loc) - np.array(emitter_loc))
                emitter_dist.append(dist)
                valid_timesteps.append(timestep)

# ✅ Total timesteps after 1st stage
print(f"✅ Total valid timesteps recorded: {len(valid_timesteps)}")


if valid_timesteps:
    plt.figure(figsize=(10, 4))
    plt.plot(valid_timesteps, emitter_dist, color='purple', label="Centroid ↔ Emitter Distance")
    plt.xlabel("Original Timestep")
    plt.ylabel("Distance")
    plt.title("Centroid Distance to Emitter Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No data recorded. Check if no_detection == 0 ever occurs and centroid/emitter locations are present.")
