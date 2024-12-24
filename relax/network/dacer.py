from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, DistributionalQNet2, DACERPolicyNet
from relax.network.common import WithSquashedGaussianPolicy
from relax.utils.diffusion import GaussianDiffusion
from relax.utils.jax_utils import random_key_from_data

class DACERParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    log_alpha: jax.Array


@dataclass
class DACERNet:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    target_entropy: float

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps)
    
    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        key, noise_key = jax.random.split(key)
        action = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        action = action + jax.random.normal(noise_key, action.shape) * jnp.exp(log_alpha) * 0.15 # other envs 0.1
        return action.clip(-1, 1)

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_alpha = policy_params
        log_alpha = -jnp.inf
        policy_params = (policy_params, log_alpha)
        return self.get_action(key, policy_params, obs)
    
    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)  # NOTE: Why not truncated normal?
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value

def create_dacer_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
) -> Tuple[DACERNet, DACERParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: DistributionalQNet2(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        log_alpha = jnp.array(1.2, dtype=jnp.float32) # 1.2 or 1.7 choose one
        return DACERParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = DACERNet(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, act_dim=act_dim, target_entropy=-act_dim*0.9)
    return net, params
