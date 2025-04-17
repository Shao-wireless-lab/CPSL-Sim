import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

matplotlib.use('TKAgg')  # Use backend for saving to file

# Change it to your local path
# Load CSV
file_path = ("/home/ece213/CPSL-Sim/results/ppo/CPSL_2025-04-10-02-30/debug/"
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
axs[3].plot(x, df["custom_metrics/agent_collisions_mean"], label="Agent Collisions", color="orange")
axs[3].plot(x, df["custom_metrics/obstacle_collisions_mean"], label="Obstacle Collisions", color="red")
axs[3].set_ylabel("Collisions (%)")
axs[3].set_title("(d) Agent & obstacle collisions")
axs[3].set_xlabel("Training Iterations")
axs[3].legend()
axs[3].grid()

# Option to save or show the plot
action = "show"  # Set to "save" to save the figure instead of displaying it
# action = "save"

output_path = "/results/plots/training.png"

if action == "save":
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📄 Figure saved to: {output_path}")
elif action == "show":
    plt.show()
else:
    print("⚠️ Invalid action. Please use 'save' or 'show'.")

