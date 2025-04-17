import json
# Change it to your local path
# Load the JSON file
with open("/home/ece213/CPSL-Sim_2/results/test_results/test_2025-03-30-19-13/trajectory_3199_1645364297.json", "r") as f:  # Replace with your actual file name
    data = json.load(f)

# Dictionaries to store collision counts per agent
agent_collisions = {}
obstacle_collisions = {}

# Iterate over timesteps
for timestep in data:
    if "infos" in timestep:
        for agent_id, agent_info in enumerate(timestep["infos"]):
            # Initialize counters if not already
            if agent_id not in agent_collisions:
                agent_collisions[agent_id] = 0
            if agent_id not in obstacle_collisions:
                obstacle_collisions[agent_id] = 0

            # Count agent-to-agent collisions
            if "agent_collision" in agent_info:
                agent_collisions[agent_id] += 1

            # Count agent-to-obstacle collisions
            if "obstacle_collision" in agent_info:
                obstacle_collisions[agent_id] += 1

# Print collision counts
print("Collision Counts:")
for agent_id in sorted(set(agent_collisions.keys()) | set(obstacle_collisions.keys())):
    print(f"Agent {agent_id}:")
    print(f"  Agent-Agent Collisions: {agent_collisions.get(agent_id, 0)}")
    print(f"  Agent-Obstacle Collisions: {obstacle_collisions.get(agent_id, 0)}")
