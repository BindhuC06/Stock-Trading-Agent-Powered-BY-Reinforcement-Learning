import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df, window_size=15, initial_balance=10000.0):
        super(StockTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
    
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        
        #updated action space for the continuous action space for the ppo
        self.action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        #The Observation Space(State)
        num_features = 7 
        obs_shape = (self.window_size * num_features) + 2
        #using -inf and inf to deal with the outliers
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_shape,), 
            dtype=np.float32
        )
        self.df = df.reset_index(drop=True)
        self.close_prices = self.df['Close'].values
        self.features_array = self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI_14']].values


    def _get_observation(self):
        window_data = self.features_array[self.current_step - self.window_size : self.current_step].copy()
        base_price = window_data[0, 3]
        raw_current_price = window_data[-1, 3]
        
        #Normalization in place 
        window_data[:, 0:4] = (window_data[:, 0:4] - base_price) / base_price
        # Normalize Volume (Col 4) via Min-Max scaling
        vol_min = np.min(window_data[:, 4])
        vol_max = np.max(window_data[:, 4])
        window_data[:, 4] = (window_data[:, 4] - vol_min) / (vol_max - vol_min + 1e-8)
        
        # Normalize SMA_20 (Col 5)
        window_data[:, 5] = (window_data[:, 5] - base_price) / base_price
        
        # Normalize RSI_14 (Col 6)
        window_data[:, 6] = window_data[:, 6] / 100.0
        
        # Flatten the 2D matrix into a 1D state vector for the neural network
        state_features = window_data.flatten()
        
        #Normalize Account Variables
        max_theoretical_balance = self.initial_balance * 3.0 
        norm_balance = self.balance / max_theoretical_balance
        
        # Calculate max possible shares using the raw, un-normalized price
        norm_shares = self.shares_held / (max_theoretical_balance / raw_current_price)
        
        # 5. Combine and cast to required Tensor format
        state = np.append(state_features, [norm_balance, norm_shares])
        
        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        return self._get_observation(), {}
        
    def step(self, action):

        # Fetch current price and calculate old portfolio value
        # We use the 'Close' price at the current step for all transactions
        current_price = self.close_prices[self.current_step]
        old_portfolio_value = self.balance + (self.shares_held * current_price)

        action_val = action[0]
        #Buy
        if action_val>0:
            # Spend a percentage of available cash based on confidence
            spend_amount=self.balance*action_val
            shares_to_buy=int(spend_amount // current_price)
            if shares_to_buy>0:
                self.shares_held+=shares_to_buy
                self.balance-=(shares_to_buy*current_price)

        # Sell Logic
        elif action_val < 0:
            # Liquidate a percentage of currently held shares based on confidence
            sell_percentage = abs(action_val)
            shares_to_sell = int(self.shares_held * sell_percentage)
            if shares_to_sell > 0:
                self.balance += (shares_to_sell * current_price)
                self.shares_held -= shares_to_sell
                
        # (Action 0 is Hold so we do nothing)
        # inc the step counter by a day
        self.current_step += 1
        # checking the termination condition. it is true when the model reaches the end of dataframe.
        done = self.current_step >= len(self.df) - 1
        
        # calculate new porfolio
        if not done:
            next_price = self.close_prices[self.current_step]
        else:
            # If done, use the last known price to evaluate final portfolio
            next_price = current_price 

        new_portfolio_value = self.balance + (self.shares_held * next_price)
        # Reward=new-old
        reward = new_portfolio_value - old_portfolio_value
        if reward<0:
            reward=reward*1.5 
        else:
            #No change
            pass
        # next state
        if not done:
            next_state = self._get_observation()
        else:
            #if its the last state then return a sate with all 0
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)

        # 7. Compile Info dictionary for debugging/logging
        info = {
            'step': self.current_step,
            'portfolio_value': new_portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held
        }
        return next_state, float(reward), done, False, info