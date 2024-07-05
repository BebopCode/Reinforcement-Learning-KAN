import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Assuming your environment and other variables are defined elsewhere
n_observations = 4
n_actions = 2

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
def initialize_env_settings(env):
    # Default values
    default_gravity = 9.8
    default_pole_mass = 0.1
    default_pole_length = 0.5
    default_cart_mass = 1.0
    default_cart_friction = 0.1
    default_pole_friction = 0.05
    default_force_mag = 10.0

    # Sample range ±10% around default values
    env.gravity = default_gravity * random.uniform(0.1, 10.0)
    env.pole_mass = default_pole_mass * random.uniform(0.1, 10.0)
    env.pole_length = default_pole_length * random.uniform(0.1, 10.0)
    env.cart_mass = default_cart_mass * random.uniform(0.1, 10.0)
    env.cart_friction = default_cart_friction * random.uniform(0.1, 10.0)
    env.pole_friction = default_pole_friction * random.uniform(0.1, 10.0)
    env.force_mag = default_force_mag * random.uniform(0.1, 10.0)
        # Print the new values
    print("New environment settings:")
    print(f"  Gravity: {env.gravity:.2f}")
    print(f"  Pole Mass: {env.pole_mass:.4f}")
    print(f"  Pole Length: {env.pole_length:.2f}")
    print(f"  Cart Mass: {env.cart_mass:.2f}")
    print(f"  Cart Friction: {env.cart_friction:.4f}")
    print(f"  Pole Friction: {env.pole_friction:.4f}")
    print(f"  Force Magnitude: {env.force_mag:.2f}")
    print()  # Add an empty line for better readability
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CartPole environment
env = gym.make('CartPole-v1')

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
# Define the structure of your policy and target networks
#policy_net = KAN([n_observations, 40, 40, n_actions]).to(device)
#target_net = KAN([n_observations, 40, 40, n_actions]).to(device)


checkpoint_path = 'rl_model_checkpoint_mlp_ii.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load the model parameters
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
target_net.load_state_dict(checkpoint['target_net_state_dict'])

# Other relevant information
episode_durations = checkpoint['episode_durations']
# Add any other information you saved if needed
num_test_episodes = 100
test_episode_durations = []
policy_net.eval()  # Set the model to evaluation mode

max_steps_per_episode = 500  # Set a maximum number of steps per episode

# Create a rendering window
env = gym.make('CartPole-v1', render_mode='human')

for i_episode in range(num_test_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    truncated = False
    t = 0
    
    initialize_env_settings(env)  # Randomize environment settings
    
    while not done and not truncated and t < max_steps_per_episode:
        env.render()  # Render the environment
        
        # Select and perform an action
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
        observation, reward, done, truncated, _ = env.step(action.item())
        
        # Move to the next state
        if not done and not truncated:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            next_state = None
        
        # Update the state
        state = next_state
        t += 1
        
    test_episode_durations.append(t)
    print(f"Episode {i_episode + 1} finished after {t} steps")

env.close()

# Calculate average episode duration
avg_duration = sum(test_episode_durations) / len(test_episode_durations)
print(f"Average episode duration over {num_test_episodes} episodes: {avg_duration:.2f}")

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(test_episode_durations)
plt.title('Episode Durations During Testing')
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.show()

# Print some statistics
print(f"Min duration: {min(test_episode_durations)}")
print(f"Max duration: {max(test_episode_durations)}")
print(f"Median duration: {sorted(test_episode_durations)[len(test_episode_durations)//2]}")