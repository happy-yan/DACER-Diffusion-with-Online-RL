from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, QNet, QScoreNet
from relax.utils.langevin import LangevinDynamics
from relax.utils.jax_utils import random_key_from_data

class QSMParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    q_score: hk.Params


@dataclass
class QSMNet:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    q_score: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int = 1

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array, *, num_particles: Optional[int] = None) -> jax.Array:
        langevin = LangevinDynamics(self.num_timesteps)
        score_params, q1_params, q2_params = policy_params
        def model_fn(x):
            return self.q_score(score_params, obs, x)

        def sample(key):
            act = langevin.sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act, q

        num_particles = num_particles if num_particles is not None else self.num_particles
        assert num_particles > 0
        if num_particles == 1:
            act = langevin.sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        else:
            keys = jax.random.split(key, num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array, *, num_particles: Optional[int] = None) -> jax.Array:
        # resample technique
        eval_num_particles = 64 if num_particles is None else num_particles
        key = random_key_from_data(obs) 
        keys = jax.random.split(key, 16)  

        def sample_best(key):
            acts = self.get_action(key, policy_params, obs, num_particles=eval_num_particles)
            acts = acts.reshape(-1, self.act_dim) 
            obs_tiled = jnp.tile(obs, (acts.shape[0], 1))  
            q1 = self.q(policy_params[1], obs_tiled, acts)
            q2 = self.q(policy_params[2], obs_tiled, acts)
            q = jnp.minimum(q1, q2)
            best_idx = jnp.argmax(q)
            return acts[best_idx]

        candidates = jax.vmap(sample_best)(keys)
        obs_tiled = jnp.tile(obs, (candidates.shape[0], 1)) 
        q1 = self.q(policy_params[1], obs_tiled, candidates)
        q2 = self.q(policy_params[2], obs_tiled, candidates)
        q = jnp.minimum(q1, q2)
        best_idx = jnp.argmax(q)
        return candidates[best_idx]

    def get_q_score_from_gradient(self, q_params: hk.Params, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        def inner(act: jax.Array, obs: jax.Array):
            # NOTE: if a special q network cannot handle unbatched inputs,
            #       we can manually unsqueeze & squeeze here
            return self.q(q_params, obs, act)
        return jax.vmap(jax.value_and_grad(inner))(act, obs)

def create_qsm_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 100,
    num_particles: int = 1,
) -> Tuple[QSMNet, QSMParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    q_score = hk.without_apply_rng(hk.transform(lambda obs, act: QScoreNet(hidden_sizes, activation)(obs, act)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, q_score_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        q_score_params = q_score.init(q_score_key, obs, act)
        return QSMParams(q1_params, q2_params, target_q1_params, target_q2_params, q_score_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = QSMNet(q=q.apply, q_score=q_score.apply, num_timesteps=num_timesteps, act_dim=act_dim, num_particles=num_particles)
    return net, params
