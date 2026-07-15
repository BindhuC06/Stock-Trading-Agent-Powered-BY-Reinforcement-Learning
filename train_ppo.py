import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env import StockTradingEnv
from stock_data import train 

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

#Monitor: to log episode rewards of SB3 properly 
env = StockTradingEnv(train, window_size=15, initial_balance=10000.0)
env = Monitor(env, "logs/")

#saving the best model  every 2000 steps
eval_callback = EvalCallback(env, 
                             best_model_save_path='./models/',
                             log_path='./logs/', 
                             eval_freq=2000,
                             deterministic=True, 
                             render=False)

#initialize ppo
model = PPO("MlpPolicy", 
            env, 
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./logs/ppo_tensorboard/")

#Train agent
print("Starting Training...")
model.learn(total_timesteps=200000, callback=eval_callback)

print("Training Complete. The best model is saved in ./models/best_model.zip")