import random
import torch
import numpy as np
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cpu"):
        """fixed-size FIFO circular buffer. capacity: Maximum number of transitions to store."""
        self.memory = deque([], maxlen=capacity)
        self.device = torch.device(device)

    def push(self, state, action, next_state, reward, done):
        """Saves a single environment transition tuple into the memory buffer."""
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int):
        """
        Uniformly samples a random mini-batch of experiences.
        Converts the batch structural elements into optimized PyTorch Tensors.
        """
        transitions = random.sample(self.memory, batch_size)
        
        # Transpose the batch from Transitions of arrays to arrays of Transitions
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        finish = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, actions, next_states, rewards, finish

    def __len__(self):
        """Returns the current number of experiences stored in the buffer."""
        return len(self.memory)
