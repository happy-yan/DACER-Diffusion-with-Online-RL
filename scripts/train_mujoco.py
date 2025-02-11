import argparse
from pathlib import Path
import time
from functools import partial

import jax, jax.numpy as jnp

from relax.algorithm.sac import SAC
from relax.algorithm.dsact import DSACT
from relax.algorithm.dacer import DACER
from relax.algorithm.qsm import QSM
from relax.algorithm.dipo import DIPO
from relax.algorithm.qvpo import QVPO
from relax.buffer import TreeBuffer
from relax.network.sac import create_sac_net
from relax.network.dsact import create_dsact_net
from relax.network.dacer import create_dacer_net
from relax.network.qvpo import create_qvpo_net
from relax.network.qsm import create_qsm_net
from relax.network.dipo import create_dipo_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience, ObsActionPair
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="dacer")
    parser.add_argument("--env", type=str, default="Humanoid-v3")
    parser.add_argument("--num_vec_envs", type=int, default=20)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(2e5)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(3e7))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    gelu = partial(jax.nn.gelu, approximate=False)

    if args.alg == "qsm":
        agent, params = create_qsm_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=args.diffusion_steps, num_particles=64)
        algorithm = QSM(agent, params, lr=args.lr)
    elif args.alg == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=args.lr)
    elif args.alg == "dsact":
        agent, params = create_dsact_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = DSACT(agent, params, lr=args.lr)
    elif args.alg == "dacer":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_dacer_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, num_timesteps=args.diffusion_steps)
        algorithm = DACER(agent, params, lr=args.lr)
    elif args.alg == "dipo":
        diffusion_buffer = TreeBuffer.from_example(
            ObsActionPair.create_example(obs_dim, act_dim),
            args.total_step,
            int(master_rng.integers(0, 2**32 - 1)),
            remove_batch_dim=False
        )
        TreeBuffer.connect(buffer, diffusion_buffer, lambda exp: ObsActionPair(exp.obs, exp.action))

        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))

        agent, params = create_dipo_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=100)
        algorithm = DIPO(agent, params, diffusion_buffer, lr=args.lr, action_gradient_steps=30, policy_target_delay=2, action_grad_norm=0.16)
    elif args.alg == "qvpo":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_qvpo_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps,
                                          num_particles=4,
                                          noise_scale=0.05)
        algorithm = QVPO(agent, params, lr=args.lr, alpha_lr=7e-3, delay_alpha_update=250)
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        evaluate_env=eval_env,
        save_policy_every=300000,
        warmup_with="random",
        log_path=PROJECT_ROOT / "logs" / args.env /
                 (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}'),
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    trainer.run(train_key)
