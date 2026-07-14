import numpy as np
import matplotlib.pyplot as plt

#Load the saved data
data = np.load('training_metrics.npz')
rewards = data['rewards']
losses = data['losses']
steps = data['steps']
epsilons = data['epsilons']

#Calculate a 10-episode rolling average for rewards
window = 10
rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DQN Trading Agent: Training Data', fontsize=16, fontweight='bold')

# Rewards -The Objective Function
axs[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
axs[0, 0].plot(range(window-1, len(rewards)), rolling_rewards, color='darkblue', linewidth=2, label=f'{window}-Ep Rolling Avg')
axs[0, 0].set_title('Episode Rewards (Profit)')
axs[0, 0].set_ylabel('Total Reward ($)')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

#Loss - The Temporal Difference Error
axs[0, 1].plot(losses, color='red', alpha=0.7)
axs[0, 1].set_title('Huber Loss (Q-Value Bootstrapping)')
axs[0, 1].set_ylabel('Smooth L1 Loss')
axs[0, 1].grid(True, alpha=0.3)

# Epsilon Decay
axs[1, 0].plot(epsilons, color='green', linewidth=2)
axs[1, 0].set_title('Exploration Rate (\u03B5-decay)')
axs[1, 0].set_ylabel('Epsilon Value')
axs[1, 0].set_xlabel('Episode')
axs[1, 0].grid(True, alpha=0.3)

#Steps per Episode
axs[1, 1].plot(steps, color='purple', alpha=0.7)
axs[1, 1].set_title('Episode Length (Data Traversal)')
axs[1, 1].set_ylabel('Steps Survived')
axs[1, 1].set_xlabel('Episode')
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_graphs.jpg')
plt.show()