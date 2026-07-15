import numpy as np
import torch
from stable_baselines3 import PPO
from env import StockTradingEnv
from stock_data import test # Import your unseen test dataset

#Test Environment
env = StockTradingEnv(test, window_size=15, initial_balance=10000.0)

#Load the Best Trained PPO Model
try:
    model = PPO.load("models/best_model.zip")
    print("Successfully loaded trained PPO model.")
except FileNotFoundError:
    raise FileNotFoundError("Could not find './models/best_model.zip'.")

# Calculate "Buy and Hold" Baseline
initial_price = test['Close'].iloc[15] 
final_price = test['Close'].iloc[-1]
shares_bought = int(10000.0 // initial_price)
leftover_cash = 10000.0 - (shares_bought * initial_price)
buy_and_hold_value = leftover_cash + (shares_bought * final_price)

#Calculate "Random Agent" Baseline (Averaged over 10 runs)
print("Simulating Continuous Random Agent Baseline...")
random_final_values = []
for _ in range(10):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample() # Generates random float between -1.0 and 1.0
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    random_final_values.append(info['portfolio_value'])

average_random_value = sum(random_final_values) / len(random_final_values)

# Evaluate the Trained PPO Agent
obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

agent_final_value = info['portfolio_value']

#Comparission
print(f"\n\nStarting Balance:         $10000.00")
print(f"\nRandom Agent (Avg Continuous): ${average_random_value:.2f}")
print(f"\nBuy & Hold Baseline:      ${buy_and_hold_value:.2f}")
print(f"\nPPO Agent Final Value:    ${agent_final_value:.2f}")

# Financial Analysis
if agent_final_value > buy_and_hold_value:
    print(f"PPO agent outperformed Buy and hold baseline by :  ${(agent_final_value - buy_and_hold_value):.2f}.")
else:
    print(f"PPO Agent underperformed the Buy and Hold Baseline by ${(buy_and_hold_value - agent_final_value):.2f}.")

if agent_final_value > average_random_value:
    print(f"The PPO agent outperformed random agent.")
else:
    print(f"The PPO agent Underperformed random agent.")
