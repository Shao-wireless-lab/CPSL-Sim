import json
import os

collision_data_path = "/home/ece213/CPSL-Sim_2/results/test_results/test_hard_60_120_2025-04-10-19-18"

# Load the JSON data from a file
with open(os.path.join(collision_data_path, "collision_seeds.json"), 'r') as f:
    data = json.load(f)

# Initialize counters
only_agent = 0
only_obstacle = 0
both = 0

# Loop through the list of collision episodes
for ep_data in data["collision_episodes"]:
    agent = ep_data.get("agent_collision", 0)
    obstacle = ep_data.get("obstacle_collision", 0)

    if agent > 0 and obstacle == 0:
        only_agent += 1
    elif agent == 0 and obstacle > 0:
        only_obstacle += 1
    elif agent > 0 and obstacle > 0:
        both += 1

# Print the results
print("Episodes with only agent collisions:", only_agent)
print("Episodes with only obstacle collisions:", only_obstacle)
print("Episodes with both agent and obstacle collisions:", both)
