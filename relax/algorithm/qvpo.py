from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.qvpo import QVPONet, QVPOParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class QVPOOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    # log_alpha: optax.OptState


class QVPOTrainState(NamedTuple):
    params: QVPOParams
    opt_state: QVPOOptStates
    step: int
    entropy: float

class QVPO(Algorithm):

    def __init__(
        self,
        agent: QVPONet,
        params: QVPOParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        lr_schedule_end = 5e-5,
        alpha_lr: float = 3e-2,
        tau: float = 0.005,
        delay_alpha_update: int = 500,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0

        self.state = QVPOTrainState(
            params=params,
            opt_state=QVPOOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: QVPOTrainState, data: Experience
        ) -> Tuple[QVPOOptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state = state.opt_state
            step = state.step
            next_eval_key, new_eval_key, diffusion_time_key, diffusion_noise_key, entropy_key = jax.random.split(
                key, 5)

            reward *= self.reward_scale

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            next_action = self.agent.get_action(next_eval_key, (policy_params, q1_params, q2_params), next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            def policy_loss_fn(policy_params, q1_params, q2_params) -> jax.Array:
                new_action = self.agent.get_action(new_eval_key, (policy_params, q1_params, q2_params), obs)
                q_mean = get_min_q(obs, new_action)
                q_weights = jnp.where(q_mean > 1., q_mean, jnp.zeros_like(q_mean))
             
                # Entropy regularization in QVPO
                ent_obs_key, ent_act_key = jax.random.split(entropy_key, 2)
                shuffled_flat_obs = jax.random.permutation(ent_obs_key, obs.flatten().repeat(10))
                rand_obs = shuffled_flat_obs.reshape((10 * obs.shape[0], obs.shape[1]))
                rand_actions = jax.random.uniform(ent_act_key, [10 * new_action.shape[0], new_action.shape[1]],
                                                  minval=-1, maxval=1)
                rand_q = jax.random.permutation(ent_obs_key, q_weights.repeat(10)) * log_alpha
                q_weights = jnp.concat([q_weights, rand_q])
                total_obs = jnp.vstack([obs, rand_obs])
                total_actions = jnp.vstack([new_action, rand_actions])
                t = jax.random.randint(diffusion_time_key, (total_obs.shape[0],), 0, self.agent.num_timesteps)

                def denoiser(t, x):
                    return self.agent.policy(policy_params, total_obs, x, t)

                loss = self.agent.diffusion.weighted_p_loss(diffusion_noise_key, q_weights, denoiser, t,
                                                            jax.lax.stop_gradient(total_actions))


                return loss, q_weights
            
            (total_loss, q_weights), policy_grads \
                = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params, q1_params, q2_params)

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def alpha_param_update(params):
                return jax.lax.cond(
                    params > 0.002,
                    lambda params: (1 - step / 2e5) * 0.02,
                    lambda params: 0.002,
                    params
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_log_alpha = alpha_param_update(log_alpha)

            state = QVPOTrainState(
                params=QVPOParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, new_log_alpha),
                opt_state=QVPOOptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state), # , log_alpha=log_alpha_opt_state
                step=step + 1,
                entropy=jnp.float32(0.0),
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1),
                "q1_max": jnp.max(q1),
                "q1_min": jnp.min(q1),
                "q2_loss": q2_loss,
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min": jnp.min(q_weights),
                "q_weights_max": jnp.max(q_weights),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.q1, self.state.params.q2 )

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
