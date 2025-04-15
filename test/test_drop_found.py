import json
import matplotlib.pyplot as plt

# Load the JSON file
with open("your_file.json", "r") as f:  # Replace with your actual file name
    data = json.load(f)

# Rewards to extract
reward_keys = ["R_task", "R_plume", "R_upwind", "R_col", "R_d", "R_theta"]

# Dictionary to store rewards for each agent
agent_rewards = {key: {} for key in reward_keys}

# Extract reward data for each agent
for timestep in data:
    if "infos" in timestep:
        for agent_id, agent_info in enumerate(timestep["infos"]):
            for key in reward_keys:
                if key in agent_info:
                    if agent_id not in agent_rewards[key]:
                        agent_rewards[key][agent_id] = []
                    agent_rewards[key][agent_id].append(agent_info[key])

# Plot rewards for each type
for key in reward_keys:
    plt.figure(figsize=(10, 6))

    for agent_id, rewards in agent_rewards[key].items():
        plt.plot(rewards, label=f"Agent {agent_id}")

    plt.xlabel("Timestep")
    plt.ylabel(f"{key} Reward")
    plt.title(f"{key} Rewards for Each Agent Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
