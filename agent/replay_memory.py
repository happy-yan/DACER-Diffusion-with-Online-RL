import numpy as np
import torch


class ReplayMemory():
    """Buffer to store environment transitions."""
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        self.states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.actions = np.empty((self.capacity, int(action_dim)), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.masks = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def append(self, state, action, reward, next_state, mask):

        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.masks[self.idx], mask)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        states = torch.as_tensor(self.states[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device)
        masks = torch.as_tensor(self.masks[idxs], device=self.device)

        return states, actions, rewards, next_states, masks


class DiffusionMemory():
    """Buffer to store best actions."""
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        self.states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.best_actions = np.empty((self.capacity, int(action_dim)), dtype=np.float32)

        self.idx = 0
        self.full = False

    def append(self, state, action):

        np.copyto(self.states[self.idx], state)
        np.copyto(self.best_actions[self.idx], action)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        states = torch.as_tensor(self.states[idxs], device=self.device)
        best_actions = torch.as_tensor(self.best_actions[idxs], device=self.device)

        best_actions.requires_grad_(True)

        return states, best_actions, idxs

    def replace(self, idxs, best_actions):
        np.copyto(self.best_actions[idxs], best_actions)
    