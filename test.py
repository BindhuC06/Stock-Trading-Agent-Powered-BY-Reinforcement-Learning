import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from Exp_replay import ReplayBuffer
from target_nets import QNetwork
from env import StockTradingEnv
from stock_data import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing the env
env = StockTradingEnv(test, window_size=15, initial_balance=10000.0)
action_counts = {0: 0, 1: 0, 2: 0} # Track Hold, Buy, Sell
num_episodes = 250
batch_size = 64
state_size = env.observation_space.shape[0]
action_size = int(env.action_space.n)

#Loading the trained model
online_net=QNetwork(state_size,action_size).to(device)

try:
    online_net.load_state_dict(torch.load('trained_agent.pth', map_location=device))
    print("Successfully loaded trained weights.")
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'trained_agent.pth'.")

online_net.eval()

#the window size is 15 and the initial amount is 10000
initial_price = test['Close'].iloc[15]
final_price = test['Close'].iloc[-1]
shares_bought = int(10000.0 // initial_price)
leftover_cash = 10000.0 - (shares_bought * initial_price)
baseline_portfolio_value = leftover_cash + (shares_bought * final_price)

random_final_values = []
for _ in range(5):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    random_final_values.append(info['portfolio_value'])

average_random_value = sum(random_final_values) / len(random_final_values)

state, info = env.reset()
done = False

while not done:
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = online_net(state_tensor)
        action = q_values.argmax().item() 
        action_counts[action] += 1

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state

agent_final_value = info['portfolio_value']

print(f"Action Distribution: {action_counts}")

# print("--- Final Evaluation Results ---\n\n")
# print("--------------------------------------------------")
# print(f"Starting Balance:       $10000.00")
# print(f"Buy & Hold Baseline:        ${baseline_portfolio_value:.2f}")
# print(f"Random Agent Final Value:       ${average_random_value:.2f}")
# print(f"DQN Agent Final Value:      ${agent_final_value:.2f}")
# print("--------------------------------------------------")

env.close()