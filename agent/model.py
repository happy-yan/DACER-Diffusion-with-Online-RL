import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.helpers import SinusoidalPosEmb, init_weights

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Model(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, time_dim=32):
        super(Model, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, time_dim),
        )

        input_dim = state_dim + action_dim + time_dim
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, action_dim))
        self.apply(init_weights)
        

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        out = torch.cat([x, t, state], dim=-1)
        out = self.layer(out)

        return out


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(MLP, self).__init__()

        input_dim = state_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Mish(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Mish())
        
        self.final_layer = nn.Linear(hidden_size, action_dim)

        self.apply(init_weights)

    def forward(self, state, eval=False, q_func=None):
        out = self.mid_layer(state)
        out = self.final_layer(out)

        if not eval:
            out += torch.randn_like(out) * 0.1

        return out

    def loss(self, action, state, weights=1.0):
        return weighted_mse_loss(self.forward(state), action, weights)

    def sample_n(self, state, times=32, chosen=1, q_func=None):
        raise NotImplemented