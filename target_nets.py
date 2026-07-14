import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim,256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)