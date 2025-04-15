import json
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')

def extract_rewards(file_path):
    """
    Reads a JSON file and extracts reward components for each agent over time.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary containing reward time-series for each agent.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)  # Load JSON file

    # Define reward keys to track
    reward_keys = ["R_task", "R_plume", "R_upwind", "R_col", "R_d", "R_theta"]

    # Dictionary to store rewards for each agent
    agent_rewards = {key: {} for key in reward_keys}

    # Extract reward data for each agent
    for iteration, entry in enumerate(data):
        if "infos" in entry:
            for agent_id, agent_data in entry["infos"].items():  # Ensure correct iteration over agents
                for key in reward_keys:
                    if key in agent_data:
                        if agent_id not in agent_rewards[key]:
                            agent_rewards[key][agent_id] = []
                        agent_rewards[key][agent_id].append(agent_data[key])
                    else:
                        if agent_id in agent_rewards[key]:  # Ensure consistent list lengths
                            agent_rewards[key][agent_id].append(0)

    # Debugging: Check extracted data
    #print("Extracted rewards:", agent_rewards)

    return agent_rewards


def plot_rewards(agent_rewards):
    """
    Plots each reward type separately for all agents over time.

    Parameters:
        agent_rewards (dict): A dictionary containing extracted reward data.
    """
    for key, agents in agent_rewards.items():
        plt.figure(figsize=(10, 6))

        no_data = True  # Flag to check if data exists

        for agent_id, rewards in agents.items():
            if rewards:  # Only plot if there's actual data
                plt.plot(rewards, label=f"Agent {agent_id}")
                no_data = False

        plt.xlabel("Timestep")
        plt.ylabel(f"{key} Reward")
        plt.title(f"{key} Rewards for Each Agent Over Time")

        if no_data:
            print(f"Warning: No data found for {key}, skipping plot.")
        else:
            plt.legend()
            plt.grid(True)
            plt.show()


# File path to the JSON file
#file_path = "/home/ece213/CPSL-Sim_2/results/test_results/test_2_2025-03-06-13-36/trajectory_3199.json"  # Update this path
file_path = "/home/ece213/CPSL-Sim_2/results/test_results/test_2025-03-05-19-14/trajectory_1227.json"
# Extract rewards
agent_rewards = extract_rewards(file_path)

# Plot rewards
plot_rewards(agent_rewards)
