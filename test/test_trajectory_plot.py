import json
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import numpy as np

matplotlib.use('TkAgg')

# Load the JSON data
with open("/home/ece213/CPSL-Sim/env_trajectory_2025-03-02-17-44_3199.json", "r") as f:
    data = json.load(f)



# Extract position data for each agent
agent_positions = {}
num_timesteps = len(data)

for timestep, timestep_data in enumerate(data):
    if "obs" in timestep_data:
        for agent_id, agent_data in timestep_data["obs"].items():
            if agent_id not in agent_positions:
                agent_positions[agent_id] = {"x": [], "y": []}
            raw_obs = agent_data.get("raw_obs", [])
            if len(raw_obs) >= 2:  # Ensure there are at least two elements (x, y)
                agent_positions[agent_id]["x"].append(raw_obs[0])
                agent_positions[agent_id]["y"].append(raw_obs[1])

# Setup figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Agent Trajectories Over Time")
ax.grid()

# Initialize plots for each agent
lines = {}   # Stores trajectory lines
scatters = {}  # Stores current position markers

for agent_id in agent_positions.keys():
    line, = ax.plot([], [], '-', label=f"Agent {agent_id}")  # Line for trajectory
    scatter = ax.scatter([], [], label=f"Agent {agent_id}", edgecolors='black', s=50)  # Current position marker
    lines[agent_id] = line
    scatters[agent_id] = scatter

ax.legend()

# Add time step text
time_text = ax.text(5, 190, "Time step: 0", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.7))

# Update function for animation
def update(frame):
    for agent_id in agent_positions.keys():
        if frame < len(agent_positions[agent_id]["x"]):
            # Get full trajectory so far
            x_traj = agent_positions[agent_id]["x"][:frame+1]
            y_traj = agent_positions[agent_id]["y"][:frame+1]

            # Update trajectory line
            lines[agent_id].set_data(x_traj, y_traj)

            # Update current position marker
            scatters[agent_id].set_offsets(np.array([[x_traj[-1], y_traj[-1]]]))

    # Update time step text
    time_text.set_text(f"Time step: {frame}")

    return list(lines.values()) + list(scatters.values()) + [time_text]

# Speed up animation with frame skipping
frame_skip = 10 # Adjust this for speed
ani = animation.FuncAnimation(fig, update, frames=range(0, num_timesteps, frame_skip), interval=50, blit=False)

# Show animation
plt.show()