import numpy as np
import torch

class ReplayBuffer:
    """Manages replay memory for training."""

    def __init__(self, max_memories, num_state_features=8):
        """
        Args:
            max_memories (int): Maximum number of memories to store.
            num_state_features (int): Number of features in the state. Defaults to 8.
        """
        self.max_memories = max_memories
        self.current_memories_ctr = 0

        # Buffers for states and next states
        self.state_memories = torch.zeros((self.max_memories, num_state_features), dtype=torch.float32)
        self.next_state_memories = torch.zeros((self.max_memories, num_state_features), dtype=torch.float32)

        # Buffer for actions
        self.action_memories = torch.zeros((self.max_memories,), dtype=torch.int32)

        # Buffer for rewards
        self.reward_memories = torch.zeros((self.max_memories,), dtype=torch.float32)

        # Buffer for terminal flags (game over)
        self.terminal_memories = torch.zeros((self.max_memories, ), dtype=torch.bool)

    def add_memories(self, state, next_state, action, reward, terminal):
        idx = self.current_memories_ctr % self.max_memories

        self.state_memories[idx] = torch.tensor(state, dtype=self.state_memories.dtype)
        self.next_state_memories[idx] = torch.tensor(next_state, dtype=self.next_state_memories.dtype)
        self.action_memories[idx] = torch.tensor(action, dtype=self.action_memories.dtype)
        self.reward_memories[idx] = torch.tensor(reward, dtype=self.reward_memories.dtype)
        self.terminal_memories[idx] = torch.tensor(terminal, dtype=self.terminal_memories.dtype)
        
        self.current_memories_ctr+=1

    def access_memory(self, batch_size, device='cpu'):
        total_memories = min(self.current_memories_ctr, self.max_memories)

        if total_memories < batch_size:
            return None

        rand_sample_idx = np.random.choice(np.arange(total_memories), size=batch_size, replace=False)
        rand_sample_idx = torch.tensor(rand_sample_idx, dtype=torch.long)

        batch = {
            "states": self.state_memories[rand_sample_idx].to(device),
            "next_states": self.next_state_memories[rand_sample_idx].to(device),
            "actions": self.action_memories[rand_sample_idx].to(device),
            "rewards": self.reward_memories[rand_sample_idx].to(device),
            "terminal": self.terminal_memories[rand_sample_idx].to(device)
        }

        return batch

