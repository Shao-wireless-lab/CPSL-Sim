import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

matplotlib.use('Agg')  # Use backend for saving to file

# Load CSV
file_path = ("/home/ece213/CPSL-Sim_2/results/ppo/CPSL_2025-04-10-02-30/debug/"
             "MyTrainer_CPSL_0v0o5_095d4_00000_0_observation_type=local,custom_model=DeepsetModel_2025-04-10_02-30-19")
df = pd.read_csv(os.path.join(file_path, "./progress.csv"))

# Set default large font size globally
plt.rcParams.update({
    "font.size": 22,            # Base font size
    "axes.titlesize": 26,       # Title size for subplots
    "axes.labelsize": 24,       # Axis label size
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22
})

# Setup large figure (good for papers)
fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
# Use training_iteration as x-axis
x = df["training_iteration"]

# Global title
fig.suptitle("Training Performance Across Time Steps", fontsize=30, y=0.94)

# (a) Mean episode rewards
axs[0].plot(x, df["episode_reward_mean"], label="Mean Reward", color="green")
axs[0].set_ylabel("Rewards")
axs[0].set_title("(a) Mean episode rewards")
axs[0].legend()
axs[0].grid()

# (b) Target found mean
if 'custom_metrics/target_found_mean' in df.columns:
    axs[1].plot(x, df['custom_metrics/target_found_mean'] * 100, label="Target Found", color="blue")
    axs[1].set_ylabel("Target Found (%)")
    axs[1].set_title("(b) Target found mean")
    axs[1].legend()
    axs[1].grid()
else:
    axs[1].text(0.5, 0.5, "Target Found data not found", ha='center', va='center', transform=axs[1].transAxes)

# (c) Final distance to emitter
axs[2].plot(x, df['custom_metrics/final_cent_dist_to_emitter_mean'], label="Mean Distance to Emitter", color="purple")
axs[2].set_ylabel("Distance (m)")
axs[2].set_title("(c) Final distance to emitter")
axs[2].legend()
axs[2].grid()

# (d) Agent and obstacle collisions
axs[3].plot(x, df["custom_metrics/agent_collisions_mean"]/8, label="Agent Collisions", color="orange")
axs[3].plot(x, df["custom_metrics/obstacle_collisions_mean"]/8, label="Obstacle Collisions", color="red")
axs[3].set_ylabel("Collisions (%)")
axs[3].set_title("(d) Agent & obstacle collisions")
axs[3].set_xlabel("Training Iterations")
axs[3].legend()
axs[3].grid()

# Layout adjustment
plt.subplots_adjust(hspace=0.5)

# Save to high-res PDF
output_path = "/home/ece213/CPSL-Sim_2/results/plots/training.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"📄 Figure saved to: {output_path}")



'''
matplotlib.use('TkAgg')

# Load the CSV file
file_path = ("/home/ece213/CPSL-Sim_2/results/ppo/CPSL_2025-04-10-02-30/debug/"
             "MyTrainer_CPSL_0v0o5_095d4_00000_0_observation_type=local,custom_model=DeepsetModel_2025-04-10_02-30-19")

df = pd.read_csv(os.path.join(file_path, "./progress.csv"))

# Function to plot episode rewards
def plot_rewards(df):
    plt.figure(figsize=(20, 6))
    plt.plot(df["num_agent_steps_trained"], df["episode_reward_max"], label="Max Reward", color="blue")
    plt.plot(df["num_agent_steps_trained"], df["episode_reward_min"], label="Min Reward", color="red")
    plt.plot(df["num_agent_steps_trained"], df["episode_reward_mean"], label="Mean Reward", color="green")
    plt.xlabel("Agent Steps Trained", fontsize=25)
    plt.ylabel("Rewards", fontsize=25)
    plt.title("Episode Rewards Over Time", fontsize=25)
    plt.legend(fontsize=20)
    plt.grid()
    plt.show(block=False)

# Function to plot target found mean if the column exists
def plot_target_found(df):
    data_name = 'custom_metrics/target_found_mean'
    if data_name in df.columns:
        plt.figure(figsize=(20, 6))
        plt.plot(df["num_agent_steps_trained"], df[data_name], label=f"{data_name}", color="purple")
        plt.xlabel("Agent Steps Trained")
        plt.ylabel("Target Found Mean")
        plt.title("Target Found Mean Over Time")
        plt.legend()
        plt.grid()
        plt.show(block=False)
    else:
        print("Column 'custom_metrics/target_found_mean' not found in the dataset.")

def plot_obstacle_collisions(df):
    data_name = "sampler_results/custom_metrics/obstacle_collisions_mean"
    if data_name in df.columns:
        plt.figure(figsize=(20, 6))
        plt.plot(df["num_agent_steps_trained"], df[data_name], label=f"{data_name}", color="purple")
        plt.xlabel("Agent Steps Trained")
        plt.ylabel(f"{data_name}")
        plt.title(f"{data_name} Over Time")
        plt.legend()
        plt.grid()
        plt.show(block=False)
    else:
        print(f"Column {data_name} not found in the dataset.")

def plot_drop_target(df):
    data_name1 = 'custom_metrics/target_found_mean'
    plt.figure(figsize=(20, 6))
    plt.plot(df["num_agent_steps_trained"], df[data_name1], label="Target found", color="blue")
    plt.xlabel("Agent Steps Trained", fontsize=25)
    plt.ylabel("Data", fontsize=25)
    plt.title("Target Found Rate Over Time", fontsize=25)
    plt.legend(fontsize=20)
    plt.grid()
    plt.show(block=False)

def plot_distance_to_emitter(df):
    data_name1 = 'custom_metrics/final_cent_dist_to_emitter_max'
    data_name2 = "custom_metrics/final_cent_dist_to_emitter_min"
    data_name3 = "custom_metrics/final_cent_dist_to_emitter_mean"
    plt.figure(figsize=(20, 6))
    plt.plot(df["num_agent_steps_trained"], df[data_name1], label="Max distance to emitter", color="blue")
    plt.plot(df["num_agent_steps_trained"], df[data_name2], label="Min distance to emitter", color="red")
    plt.plot(df["num_agent_steps_trained"], df[data_name3], label="Mean distance to emitter", color="green")
    plt.xlabel("Agent Steps Trained", fontsize=25)
    plt.ylabel("Distance", fontsize=25)
    plt.title("Final Distance to Emitter Over Time", fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(which='both', linestyle='--', alpha=0.7)
    plt.show(block=False)

def plot_agent_collision(df):
    data_name1 = 'custom_metrics/agent_collisions_max'
    data_name2 = "custom_metrics/agent_collisions_min"
    data_name3 = "custom_metrics/agent_collisions_mean"
    plt.figure(figsize=(20, 6))
    plt.plot(df["num_agent_steps_trained"], df[data_name1], label="Max agent collisions", color="blue")
    plt.plot(df["num_agent_steps_trained"], df[data_name2], label="Min agent collisions", color="red")
    plt.plot(df["num_agent_steps_trained"], df[data_name3], label="Mean agent collisions", color="green")
    plt.xlabel("Agent Steps Trained", fontsize=25)
    plt.ylabel("Collisions", fontsize=25)
    plt.title("Agent Collisions Over Time", fontsize=25)
    plt.legend(fontsize=20)
    plt.grid()
    plt.show(block=False)

def plot_obstacles_collision(df):
    data_name1 = 'custom_metrics/obstacle_collisions_max'
    data_name2 = "custom_metrics/obstacle_collisions_min"
    data_name3 = "custom_metrics/obstacle_collisions_mean"
    plt.figure(figsize=(20, 6))
    plt.plot(df["num_agent_steps_trained"], df[data_name1], label="Max obstacle collisions", color="blue")
    plt.plot(df["num_agent_steps_trained"], df[data_name2], label="Min obstacle collisions", color="red")
    plt.plot(df["num_agent_steps_trained"], df[data_name3], label="Mean obstacle collisions", color="green")
    plt.xlabel("Agent Steps Trained", fontsize=25)
    plt.ylabel("Collisions", fontsize=25)
    plt.title("Obstacle Collisions Over Time", fontsize=25)
    plt.legend(fontsize=20)
    plt.grid()
    plt.show(block=False)

# Call plotting functions
plot_rewards(df)
plot_drop_target(df)
plot_distance_to_emitter(df)
plot_agent_collision(df)
plot_obstacles_collision(df)

# Optional
# plot_target_found(df)
# plot_obstacle_collisions(df)

plt.show()
'''