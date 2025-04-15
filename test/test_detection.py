import json

# Load the JSON file
with open("/home/ece213/CPSL-Sim_2/results/test_results/test_2025-03-27-17-06/trajectory_3199.json", "r") as f:
    data = json.load(f)

# To store the time steps where no_detection == 0
no_detection_timesteps = []

# Iterate over timesteps
for timestep, entry in enumerate(data):
    if "infos" in entry:
        for agent_id, agent_info in entry['infos'].items():
            if "no_detection" in agent_info and agent_info["no_detection"] == 0:
                print(f"Gas first detected at timestep {timestep}!")
                # Exit both loops immediately
                exit()