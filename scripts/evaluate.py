import glob
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

from tensorboardX import SummaryWriter
from pathlib import Path
import argparse
import itertools
import pickle

import numpy as np
import jax

from relax.env import create_env
from relax.utils.persistence import PersistFunction

def iter_key(key: jax.Array):
    def iter_key_fn(step: int):
        return jax.random.fold_in(key, step)

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up

    for i in itertools.count():
        yield iter_key_fn(i)
        
def list_policy_files_by_creation_time(directory):
    policy_files = glob.glob(os.path.join(directory, 'policy*'))
    policy_files.sort(key=lambda x: os.path.getctime(x))
    return policy_files
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=Path, default=Path("/home/xxx/policy-28200000-1410000.pkl"))
    parser.add_argument("--env", type=str, default="Ant-v3")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--stochastic", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    with open(args.policy_path, "rb") as f:
        policy_params = pickle.load(f)
        policy_params, log_alpha, q1, q2 = policy_params
        log_alpha[...] = -np.inf
        policy_params = policy_params, log_alpha, q1, q2
        
    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2**32 - 1, 3))
    env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    policy_iter_key = iter_key(jax.random.key(policy_seed))
    
    @jax.jit
    def policy_fn(key, obs):
        if args.stochastic:
            policy = PersistFunction.load(args.policy_path.with_name("stochastic.pkl"))
            return policy(key, policy_params, obs).clip(-1, 1)
        else:
            policy = PersistFunction.load(args.policy_path.with_name("deterministic.pkl"))
            return policy(policy_params, obs).clip(-1, 1)
    
    @jax.jit
    def logp_fn(key, obs):
        policy = PersistFunction.load(args.policy_path.with_name("logp.pkl"))
        action, logp = policy(key, policy_params, obs)
        return action.clip(-1, 1), logp
        
    # 定义文件名
    csv_file = 'obs_data.csv'
    import csv
    
    # 打开文件准备写入数据
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)  
        total_reward_list = []
        for i in range(args.num_episodes):
            obs, info = env.reset()
            total_reward = 0.0
            done = False
            count = 0
            while not done:
                # with jax.disable_jit():
                action = jax.device_get(policy_fn(next(policy_iter_key), obs))
                a, logp = jax.device_get(logp_fn(next(policy_iter_key), obs))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                count += 1
                writer.writerow(obs.tolist()) 
            total_reward_list.append(total_reward)
            print(f"Episode {i+1}: Total Reward = {total_reward}")
        
    value = np.mean(total_reward_list)
    env.close()
