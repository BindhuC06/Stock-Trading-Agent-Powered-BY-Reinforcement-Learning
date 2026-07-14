import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from Exp_replay import ReplayBuffer
from target_nets import QNetwork
from env import StockTradingEnv
from stock_data import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing the env
env = StockTradingEnv(train, window_size=15, initial_balance=10000.0)

num_episodes = 250
batch_size = 64
state_size = env.observation_space.shape[0]
action_size = int(env.action_space.n)

max_reward = -float('inf') 
buffer = ReplayBuffer(capacity=50000, device=device)

online_net = QNetwork(state_size, action_size).to(device)
target_net = QNetwork(state_size, action_size).to(device)
target_net.load_state_dict(online_net.state_dict())

optimizer = optim.Adam(online_net.parameters(), lr=1e-4)
loss_fn = nn.SmoothL1Loss()
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99 
'''We will be using the soft update over the target network. 
Tau is a parameter for polyak averaging.'''
tau = 0.005

def soft_update(online, target, tau):
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

step = 0
rewards_list, loss_list, steps_list, epsilon_list = [], [], [], []

for ep in range(1, num_episodes + 1):
    state, info = env.reset()
    done = False
    ep_step = 0
    total_reward = 0
    episode_losses = []
    
    while not done:
        online_net.train()
        
        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = online_net(state_tensor)
                action = q_values.argmax().item()

        #next state
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        buffer.push(state, action, next_state, reward, done)
        state = next_state
        total_reward += reward
        step += 1
        ep_step += 1
        
        # Training
        if len(buffer) >= batch_size:
            states, actions, next_states, rewards, dones = buffer.sample(batch_size)
            current_q_values = online_net(states).gather(1, actions)
            
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(1, keepdim=True)[0]
                target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
            
            loss = loss_fn(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping to prevent weight explosion from high financial rewards
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
            
            optimizer.step()
            episode_losses.append(loss.item())
            
            soft_update(online_net, target_net, tau)
            
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # log the reward, steps and epsilon value. these will be useful while debugging
    rewards_list.append(total_reward)
    steps_list.append(ep_step)
    epsilon_list.append(epsilon)
    
    avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
    loss_list.append(avg_loss)
    
    max_reward = max(max_reward, total_reward)
    
    if ep % 10 == 0:
        print(f"Episode {ep} | Reward: ${total_reward:.2f} | Final Portfolio: ${info['portfolio_value']:>8.2f} | Loss: {avg_loss:>6.4f} | Epsilon: {epsilon:.2f}")

np.savez('training_metrics.npz', 
         rewards=rewards_list, 
         losses=loss_list, 
         steps=steps_list, 
         epsilons=epsilon_list)

print(f"\nTraining Complete. Maximum Episode Reward: ${max_reward:.2f}")

env.close()