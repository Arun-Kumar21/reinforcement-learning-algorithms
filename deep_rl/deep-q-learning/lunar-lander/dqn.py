import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-Network with 3 fully connected layers and ReLU activations."""
    def __init__(self,
                 input_state=8,
                 hidden_units=128,
                 num_actions=4):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_state, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, num_actions)

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
        
