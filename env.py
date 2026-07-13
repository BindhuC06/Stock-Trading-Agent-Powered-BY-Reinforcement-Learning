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
        
        # The Action Space- 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # The Observation Space(State)
        num_features = 7 
        obs_shape = (self.window_size * num_features) + 2
        #using -inf and inf to deal with the outliers
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_shape,), 
            dtype=np.float32
        )
    def _get_observation(self):
        window_df = self.df.iloc[self.current_step - self.window_size : self.current_step]
        # This prevents the NN from seeing absolute prices, only relative growth
        base_price = window_df['Close'].iloc[0]
        normalized_window = window_df.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            normalized_window[col] = (window_df[col] - base_price) / base_price
        vol_min = window_df['Volume'].min()
        vol_max = window_df['Volume'].max()
        normalized_window['Volume'] = (window_df['Volume'] - vol_min) / (vol_max - vol_min + 1e-8) # 1e-8 prevents division by zero
        
        normalized_window['SMA_20'] = (window_df['SMA_20'] - base_price) / base_price
        # RSI is strictly 0 to 100 hence we do direct division
        normalized_window['RSI_14'] = window_df['RSI_14'] / 100.0
        state_features = normalized_window.values.flatten()
        
        # Normalize Account Information
        # Assuming a theoretical max balance to scale between 0 and 1
        max_theoretical_balance = self.initial_balance * 3 
        norm_balance = self.balance / max_theoretical_balance
        
        # Assuming max shares we could possibly own
        max_shares = max_theoretical_balance / self.df['Close'].iloc[self.current_step]
        norm_shares = self.shares_held / max_shares
        
        # 4. Combine into final state vector
        state = np.append(state_features, [norm_balance, norm_shares])
        
        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        return self._get_observation(), {}
        
    def step(self, action):

        '''Please note that this model is designed such that it will buy however many shares that can be 
        bought with the current amount/balance and it will sell everything at once as well.'''
        # Fetch current price and calculate old portfolio value
        # We use the 'Close' price at the current step for all transactions
        current_price = self.df.iloc[self.current_step]['Close']
        old_portfolio_value = self.balance + (self.shares_held * current_price)

        #  action execution
        if action == 1:  # Buy
            # Calculate maximum whole shares we can afford
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= (shares_to_buy * current_price)
                
        elif action == 2:  # Sell
            # Liquidate all held shares
            if self.shares_held > 0:
                self.balance += (self.shares_held * current_price)
                self.shares_held = 0
                
        # (Action 0 is Hold so we do nothing)

        # inc the step counter by a day
        self.current_step += 1
        
        # checking the termination condition. it is true if the model reached the end of dataframe.
        done = self.current_step >= len(self.df) - 1
        # calculate new porfolio
        if not done:
            next_price = self.df.iloc[self.current_step]['Close']
        else:
            # If done, use the last known price to evaluate final portfolio
            next_price = current_price 

        new_portfolio_value = self.balance + (self.shares_held * next_price)
        
        # Reward=new-old
        reward = new_portfolio_value - old_portfolio_value

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
        terminated = done
        truncated = False

        return next_state, reward, terminated, truncated, info