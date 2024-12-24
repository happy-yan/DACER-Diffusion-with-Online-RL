import torch
from torch import nn
from torch.nn import functional as F
from agent.helpers import init_weights

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_size=256,
                 behavior_sample=16, eval_sample=512, deterministic=False) -> None:
        super(VAE, self).__init__()

        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.behavior_sample = behavior_sample
        self.eval_sample = eval_sample
        self.deterministic = deterministic

        input_dim = state_dim + action_dim

        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                     nn.Mish(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.Mish(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.Mish())

        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_var = nn.Linear(hidden_size, hidden_size)

        self.decoder = nn.Sequential(nn.Linear(hidden_size + state_dim, hidden_size),
                                     nn.Mish(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.Mish(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.Mish())

        self.final_layer = nn.Sequential(nn.Linear(hidden_size, action_dim))

        self.apply(init_weights)

        self.device = device

    def encode(self, action, state):
        x = torch.cat([action, state], dim=-1)
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z, state):
        x = torch.cat([z, state], dim=-1)
        result = self.decoder(x)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss(self, action, state, weights=1.0):
        mu, log_var = self.encode(action, state)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z, state)

        kld_weight = 0.1  # Account for the minibatch samples from the dataset
        recons_loss = weighted_mse_loss(recons, action, weights)

        kld_loss = torch.mean(-0.5 * torch.sum((1 + log_var - mu ** 2 - log_var.exp()) * weights, dim=1), dim=0)

        # print('recons_loss: ', recons_loss)
        # print('kld_loss: ', kld_loss)

        loss = recons_loss + kld_weight * kld_loss
        return loss

    def forward(self, state, eval=False, q_func=None):
        batch_size = state.shape[0]
        shape = (batch_size, self.hidden_size)

        if eval:
            z = torch.zeros(shape, device=self.device)
        else:
            z = torch.randn(shape, device=self.device)
        samples = self.decode(z, state)

        return samples.clamp(-1., 1.)

    def sample_n(self, state, times=32, chosen=1, q_func=None):
        old_state = state
        raw_batch_size = state.shape[0]
        state = state.repeat(times, 1)
        action = self(state)
        action.clamp_(-1., 1.)
        q1, q2 = q_func(state, action)
        q = torch.min(q1, q2)
        action = action.view(times, raw_batch_size, -1).transpose(0, 1)
        q = q.view(times, raw_batch_size, -1).transpose(0, 1)
        if chosen == 1:
            action_idx = torch.argmax(q, dim=1, keepdim=True).repeat(1, 1, self.action_dim)
            return old_state, action.gather(dim=1, index=action_idx).view(raw_batch_size, -1)
        else:
            action_idx = torch.topk(q, k=chosen, dim=1)[1].repeat(1, 1, self.action_dim)
            return old_state.repeat(chosen, 1).view(chosen, raw_batch_size, -1).transpose(0,1).contiguous().view(raw_batch_size*chosen, -1), action.gather(dim=1, index=action_idx).view(raw_batch_size*chosen, -1)
