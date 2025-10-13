import torch.nn as nn
import torch
import torch.optim as optim

import numpy as np

from dqn import DQN
from replay_buffer import ReplayBuffer

class Agent:
    def __init__(self,
                 num_actions,
                 max_memories=100_000,
                 gamma=0.99,
                 lr=0.001,
                 input_state_features=8,
                 hidden_features=128,
                 epsilon=1.0,
                 epsilon_decay=0.999,
                 min_epsilon=0.05,
                 device='cpu'):

        self.num_actions=num_actions
        self.max_memories = max_memories
        self.gamma=gamma
        self.lr=lr
        self.input_state_features=input_state_features
        self.hidden_features=hidden_features
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.min_epsilon=min_epsilon
        self.device=device

        # DQ Network
        self.DQN=DQN(self.input_state_features, self.hidden_features).to(self.device)

        # Initialize the target network to stabilize training by addressing the moving target problem
        self.DQN_NEXT=DQN(self.input_state_features, self.hidden_features).to(device)
        self.DQN_NEXT.load_state_dict(self.DQN.state_dict())
        self.DQN_NEXT.eval()

        # Adam Optimizer
        self.optimizer=optim.Adam(self.DQN.parameters(), lr=self.lr)

        # MSE Loss
        self.loss_fn=nn.MSELoss()

        self.replay_buffer=ReplayBuffer(self.max_memories, self.input_state_features)

    
    def select_action(self, state):

        if not isinstance(state, torch.Tensor):
            state=torch.tensor(state, device=self.device)

        # Create batch dim if not
        if state.dim() == 1:
            state=state.unsqueeze(0)

        assert state.shape[-1]==self.input_state_features, f"Passing {state.shape[-1]} features, expect {self.input_state_features}"

        # Epsilon Greedy selection
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)

        else:
            self.DQN.eval()
            with torch.no_grad():
                Q_sa=self.DQN(state)
            action=torch.argmax(Q_sa).item()
            self.DQN.train()

        return action

    def update_epsilon(self):
        self.epsilon=max(self.min_epsilon, self.epsilon*self.epsilon_decay)

    # Function to update Target network after few steps
    def update_target_network(self):
        self.DQN_NEXT.load_state_dict(self.DQN.state_dict())

    def inference(self, state, device='cpu'):
        self.DQN = self.DQN.to(device)
        self.DQN.eval()

        with torch.no_grad():
            Q_sa=self.DQN(state.to(device))

        return torch.argmax(Q_sa).item()


    def train_step(self, batch_size):
        batch=self.replay_buffer.access_memory(batch_size, self.device)

        if batch is None:
            return None

        # (B x Actions)
        q_estimate=self.DQN(batch["states"])
        
        q_estimate=torch.gather(q_estimate, index=batch["actions"].unsqueeze(-1), dim=-1).squeeze(-1)
      
        with torch.no_grad():
            self.DQN.eval()
            # Use Target Network
            q_next_estimate=self.DQN_NEXT(batch["next_states"])
            self.DQN.train()
    
        max_q_estimate=torch.max(q_next_estimate, dim=-1).values
    
        td_target=batch["rewards"] + self.gamma*max_q_estimate * (~batch["terminal"])

            
        # We want to move our model outputs (q_estimate) close to TD targets
        loss=self.loss_fn(td_target, q_estimate)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Updating epsilon
        self.update_epsilon()

