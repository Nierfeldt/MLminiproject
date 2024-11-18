import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Fixed 11x11 custom map
custom_map = [
    "SFFFFFHFFFF",
    "HFFFFFFFFFF",
    "FFFHFFFFFFF",
    "FFFFFFFFFFF",
    "FFFFHFFFFFH",
    "FFFFFFFFHFF",
    "FFFHFFFFFFF",
    "FFFFFHFFFFF",
    "FFFFFFFFFFF",
    "HFFFFFFFFFF",
    "FFFFFFFFFFG"
]

# Create Frozen Lake environment
env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")

# Initialize the Q-table
state_space = env.observation_space.n
action_space = env.action_space.n
q_table = np.zeros((state_space, action_space))

# Hyperparameters
alpha = 0.8  # Learning rate (can stay high for faster convergence)
gamma = 0.95  # Discount factor (retains focus on long-term rewards)
epsilon = 0.1  # Initial exploration rate
epsilon_decay = 0.1  # Slower decay for more exploration
min_epsilon = 0.1  # Minimum exploration rate
episodes = 5000  # Increased episodes for better learning on large maps
max_steps = 500  # Maximum steps per episode

episode_rewards = []
states_visited = []
actions_taken = []

# Load Q-table if it exists
try:
    q_table = np.load("frozenlake_qtable.npy")
    print("Loaded saved Q-table from 'frozenlake_qtable.npy'.")
except FileNotFoundError:
    print("No saved Q-table found. Starting fresh.")

# Specify the output directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        # Choose action using epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit

        next_state, reward, done, _, _ = env.step(action)

        goal_state = state_space - 1  # Assuming the goal is the last state
        distance_to_goal = abs(state - goal_state)
        next_distance_to_goal = abs(next_state - goal_state)

        if done and reward == 0:
            reward = -100  # Penalize falling into a hole
        elif done and reward == 1:
            reward = 100  # Reward for reaching the goal
        else:
            step_penalty = -10  # Small penalty for each step taken
            reward = step_penalty + 10 * (distance_to_goal - next_distance_to_goal)

        # Q-learning update
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )

        # Record states and actions
        states_visited.append(state)
        actions_taken.append(action)

        state = next_state
        total_rewards += reward

        if done:
            break

    # Decay exploration rate
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(total_rewards)

    # Debug log every 500 episodes
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1} | Epsilon: {epsilon:.4f} | Last Reward: {total_rewards}")

# Save the trained Q-table
np.save("frozenlake_qtable.npy", q_table)
print("Trained Q-table saved to 'frozenlake_qtable.npy'.")

# Testing the agent
test_episodes = 100
total_rewards = 0
for episode in range(test_episodes):
    state, _ = env.reset()
    for step in range(max_steps):
        action = np.argmax(q_table[state, :])  # Exploit
        next_state, reward, done, _, _ = env.step(action)
        total_rewards += reward
        state = next_state
        if done:
            break

print(f"Average reward over {test_episodes} test episodes: {total_rewards / test_episodes}")


# Helper Functions
def qtable_directions_map(q_table, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = q_table.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(q_table, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(q_table, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(q_table, map_size)

    # Capture the last frame of the environment
    env.reset()
    frame = env.render()  # Render mode should be set to 'rgb_array'

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(frame)
    ax[0].axis("off")
    ax[0].set_title("Last Frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    plt.tight_layout()
    
    # Instead of plt.show(), save the figure
    plot_filename = os.path.join(output_dir, "q_values_map.png")
    plt.savefig(plot_filename)
    plt.close()


def plot_states_actions_distribution(states, actions):
    """Plot the distributions of states and actions."""
    labels = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Plot distribution of states
    sns.histplot(data=states, ax=ax[0], kde=True, bins=state_space)
    ax[0].set_title("State Distribution")

    # Plot distribution of actions
    sns.histplot(data=actions, ax=ax[1], bins=4)
    ax[1].set_xticks(list(labels.keys()), labels=labels.values())
    ax[1].set_title("Action Distribution")

    plt.tight_layout()
    
    # Instead of plt.show(), save the figure
    plot_filename = os.path.join(output_dir, "states_actions_distribution.png")
    plt.savefig(plot_filename)
    plt.close()


# Visualize the Q-table and policy
plot_q_values_map(q_table, env, len(custom_map))

# Visualize state and action distributions
plot_states_actions_distribution(states_visited, actions_taken)

# Visualize the Q-table
print("Trained Q-Table:")
print(q_table)