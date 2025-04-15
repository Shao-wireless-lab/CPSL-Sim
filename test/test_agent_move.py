import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from cuas.agents.cuas_agents import Agent, AgentType
import matplotlib
from datetime import datetime

matplotlib.use('TkAgg')


# Test function to check agent's movement including U-turn
def test_agent_step():
    # Initialize agent at (10, 10) with theta=0, radius=1, and observation radius=20
    agent = Agent(x=np.int32(10), y=np.int32(10), theta=0, r=1, obs_r=20, type2=AgentType.P)

    # Define a set of actions: (velocity, angular velocity)
    actions = [
        (10, 0),          # Move forward straight
        (10, 0),          # Move forward straight
        "u_turn",         # Simulate a U-turn by flipping theta
        (10, 0),          # Move forward in new direction
        (10, 0)           # Move forward
    ]

    print("Initial state:", agent.state)

    for i, action in enumerate(actions):
        if action == "u_turn":
            agent.theta = (agent.theta + math.pi) % (2 * math.pi)
            print(f"State after U-turn {i + 1}: {agent.state}")
        else:
            agent.step(action)
            print(f"State after action {i + 1}: {agent.state}")


# Animation function to visualize movement including U-turn
def animate_agent():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_title("Agent Movement Animation with U-Turn")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    v_max = 150
    agent = Agent(x=np.int32(10), y=np.int32(10), theta=0, r=1, obs_r=20, type2=AgentType.P)

    # Actions with a U-turn inserted
    actions = [
        (v_max/6, 0),
        (v_max/6, 0),
        (v_max/6, 0),
        "u_turn",               # <-- U-TURN here
        (v_max/6, 0),
        (v_max/6, 0),
        (v_max/6, 0),
        (v_max/6, 0)
    ]

    trajectory_x = [agent.x]
    trajectory_y = [agent.y]

    agent_marker, = ax.plot([], [], 'bo', markersize=10)
    trajectory_line, = ax.plot([], [], 'r-', linewidth=2)

    def init():
        agent_marker.set_data([], [])
        trajectory_line.set_data([], [])
        return agent_marker, trajectory_line

    def update(frame):
        if frame < len(actions):
            action = actions[frame]
            if action == "u_turn":
                agent.theta = (agent.theta + math.pi) % (2 * math.pi)
            else:
                agent.step(action)

            trajectory_x.append(agent.x)
            trajectory_y.append(agent.y)

        agent_marker.set_data([agent.x], [agent.y])
        trajectory_line.set_data(trajectory_x, trajectory_y)

        return agent_marker, trajectory_line

    ani = animation.FuncAnimation(fig, update, frames=len(actions)+5, init_func=init, blit=True, interval=300)
    plt.show()


if __name__ == "__main__":
    test_agent_step()
    animate_agent()
