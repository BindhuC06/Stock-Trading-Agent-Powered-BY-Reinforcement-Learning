# Stock-Trading-Agent-Powered-BY-Reinforcement-Learning
Deep Q-Network trading agent for single-stock trading, built with PyTorch and Gymnasium-style custom environment. Uses price history and technical indicators like- SMA, RSI as state, evaluated against a buy-and-hold baseline on held-out test data.

# Deep Reinforcement Learning for Algorithmic Trading

This project explores the application of Deep Reinforcement Learning (DRL) to algorithmic trading. The objective was to train an autonomous agent to navigate a custom historical stock market environment (Gymnasium) and maximize portfolio value. In this case the stock taken was JEPQ form yfinance.

The architecture underwent two distinct evolutionary phases:
1)**Deep Q-Network (DQN)** and shifted to
2)**Proximal Policy Optimization (PPO)** Actor-Critic model to solve inherent mathematical bottlenecks 
in initial action spaces.

---

## Phase 1 Evaluation: DQN and the Discrete Action Space Limitations

The initial architecture used a Deep Q-Network (DQN), similar to the one used in cartpole, 
with a discrete action space `[0: Hold, 1: Buy, 2: Sell]`. 
While the network successfully converged during training (Max Reward ~$13,000), it is constantly underperforming the Buy & Hold baseline and a random agent on unseen test data. Logging of the agent's action distribution after evaluation revealed that the agent had fallen into a bad policy of hyperactive "panic selling," proving that discrete action spaces are fundamentally brittle for real-world financial environments.

### Why the DQN is failing?

1.  **The "All-In/All-Out" Constraint:** Discrete action spaces and environments forced the agent to either use 100% of its current cash to buy the shares or sell (liquidate) 100% of its shares which made the agent unable to position the cash and understand the stock. 
2.  **Reward Shaping Conflict:** To prevent overfitting to market noise, a 2x penalty on negative portfolio changes was introduced. However, combined with the discrete action space, this totally backfired. If the 15-day SMA/RSI window predicted even a minor, temporary dip, the network calculated the 2x penalty and executed a full portfolio liquidation to protect itself from loss.
3.  **Policy Oscillation:** Action distribution logs during testing (`{0: 12, 1: 87, 2: 32}`) showed that the agent executed a 100% portfolio liquidation 32 times across 131 trading days which is roughly once every 4 days. The agent was confused by market noise, constantly buying high and panic-selling low at the slightest hint of volatility.

### Phase 2: Proximal Policy Optimization (PPO) & Continuous Action space
To solve the discrete bottleneck, the architecture was migrated to a continuous action space $A \in [-1.0, 1.0]$ using Stable-Baselines3's PPO implementation. 

* **Feature Implementation:**
  * Continuous portfolio sizing: An action of `0.5` spends 50% of available cash; `-0.2` liquidates 20% of held shares. This allowed the agent to scale confidence and reduce risk without full liquidation.
  * Rolling Evaluation Checkpoints via SB3 `EvalCallback` to mathematically guarantees that the saved model is highly competent (the best one during training) and not a random last epoch save.

---

## Evaluation & Results

The final PPO model was evaluated deterministically on an unseen 20% hold-out test dataset against two strict baselines.

| Strategy | Final Portfolio Value | Result |
| :--- | :--- | :--- |
| **Buy & Hold** | $10,937.93 | Benchmark |
| **PPO Agent** | $10,243.67 | Underperformed Baseline |
| **Random Agent** | $10,237.30 | **Outperformed Random** |

### What went wrong?
The PPO agent successfully bypassed the random noise floor, proving the Actor-Critic network learned to extract valid info and avoid the policy oscillation of the previous DQN. 

However, it underperformed the Buy & Hold strategy. In a strong, unidirectional market, Buy & Hold absorbs 100% of the upside. Because the PPO agent was trained to prioritize capital preservation (via the Asymmetric Risk Penalty, the 1.5 time loss), it actively hedged its positions, sacrificing maximum absolute profit to protect against downside volatility. 

---

## Current Limitations & Future Work

While the continuous architecture is mathematically sound, the agent's predictive ceiling is currently constrained by the environment.

1. **Information Theory Constraints:** The state space relies on raw OHLCV and lagging indicators (SMA, RSI). These features describe past states rather than predicting future volatility.
   * *Fix:* Expand feature engineering to include leading momentum oscillators (MACD), volatility bands (Bollinger Bands/ATR), and macro-market indices (e.g., SPY) to provide broader market context.

2. **Absolute Profit Objective:** The environment currently rewards the agent based strictly on absolute portfolio delta ($\Delta V_t$). 
   * *Fix:* Rewrite the step function reward to optimize for **Risk-Adjusted Return**.

* *note* :- This Project was made as a part of 30 day self-study reinforecement learning roadmap. This Capstone Project marks the end of this track.
