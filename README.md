# Stock-Trading-Agent-Powered-BY-Reinforcement-Learning
Deep Q-Network trading agent for single-stock trading, built with PyTorch and Gymnasium-style custom environment. Uses price history and technical indicators like- SMA, RSI as state, evaluated against a buy-and-hold baseline on held-out test data.
## Phase 1 Evaluation: DQN and the Discrete Action Space Limitations

The initial architecture used a Deep Q-Network (DQN), similar to the one used in cartpole, 
with a discrete action space `[0: Hold, 1: Buy, 2: Sell]`. 
While the network successfully converged during training (Max Reward ~$13,000), it is constantly underperforming the Buy & Hold baseline and a random agent on unseen test data. Logging of the agent's action distribution after evaluation revealed that the agent had fallen into a bad policy of hyperactive "panic selling," proving that discrete action spaces are fundamentally brittle for real-world financial environments.

### Why the DQN is failing?

1.  **The "All-In/All-Out" Constraint:** Discrete action spaces and environments forced the agent to either use 100% of its current cash to buy the shares or sell (liquidate) 100% of its shares which made the agent unable to position the cash and understand the stock. 
2.  **Reward Shaping Conflict:** To prevent overfitting to market noise, a 2x penalty on negative portfolio changes was introduced. However, combined with the discrete action space, this totally backfired. If the 15-day SMA/RSI window predicted even a minor, temporary dip, the network calculated the 2x penalty and executed a full portfolio liquidation to protect itself from loss.
3.  **Policy Oscillation:** Action distribution logs during testing (`{0: 12, 1: 87, 2: 32}`) showed that the agent executed a 100% portfolio liquidation 32 times across 131 trading days which is roughly once every 4 days. The agent was confused by market noise, constantly buying high and panic-selling low at the slightest hint of volatility.

### Phase 2 Architecture: Continuous Action Spaces (PPO)
To resolve this, the architecture must transition away from Q-Learning to an Actor-Critic algorithm like **Proximal Policy Optimization (PPO)**. By outputting a continuous float $A \in [-1, 1]$, the agent will be able to dynamically allocate portfolio percentages (e.g., selling only 20% of a position during a minor dip), closely mirroring real-world algorithmic trading strategies.