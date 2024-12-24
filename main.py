import argparse
import copy

import numpy as np
import torch

from agent.qvpo import QVPO
from agent.replay_memory import ReplayMemory, DiffusionMemory

from tensorboardX import SummaryWriter
import gym
import os
from logger import Logger
import datetime
from tqdm import tqdm
from vector_env import VectorEnv
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy')
    parser.add_argument('--env_name', default="Hopper-v3",
                        help='Mujoco Gym environment (default: Hopper-v3)')
    parser.add_argument('--seed', type=int, default=200, metavar='N',
                        help='random seed (default: 0)')

    parser.add_argument('--num_steps', type=int, default=1500000, metavar='N',
                        help='env timesteps (default: 1000000)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int, default=1, metavar='N',
                        help='update actor target per iteration (default: 1)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=20, metavar='N',
                        help='diffusion timesteps (default: 20)')
    parser.add_argument('--diffusion_lr', type=float, default=0.0001, metavar='G',
                        help='diffusion learning rate (default: 0.0001)')
    parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                        help='critic learning rate (default: 0.0003)')
    parser.add_argument('--action_lr', type=float, default=0.03, metavar='G',
                        help='diffusion learning rate (default: 0.03)')
    parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                        help='noise ratio in sample process (default: 1.0)')

    parser.add_argument('--action_gradient_steps', type=int, default=20, metavar='N',
                        help='action gradient steps (default: 20)')
    parser.add_argument('--ratio', type=float, default=0.1, metavar='G',
                        help='the ratio of action grad norm to action_dim (default: 0.1)')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                        help='actor and critic grad norm (default: 1.0)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')

    parser.add_argument('--alpha_mean', type=float, default=0.001, metavar='G',
                        help='running mean update weight (default: 0.1)')

    parser.add_argument('--alpha_std', type=float, default=0.001, metavar='G',
                        help='running std update weight (default: 0.001)')

    parser.add_argument('--beta', type=float, default=1.0, metavar='G',
                        help='expQ weight (default: 1.0)')

    parser.add_argument('--weighted', action="store_true", help="weighted training")

    parser.add_argument('--aug', action="store_true", help="augmentation")

    parser.add_argument('--train_sample', type=int, default=64, metavar='N',
                        help='train_sample (default: 64)')

    parser.add_argument('--chosen', type=int, default=1, metavar='N', help="chosen actions (default:1)")

    parser.add_argument('--q_neg', type=float, default=0.0, metavar='G', help="q_neg (default: 0.0)")

    parser.add_argument('--behavior_sample', type=int, default=4, metavar='N', help="behavior_sample (default: 1)")
    parser.add_argument('--target_sample', type=int, default=4, metavar='N', help="target_sample (default: behavior sample)")

    parser.add_argument('--eval_sample', type=int, default=32, metavar='N', help="eval_sample (default: 512)")

    parser.add_argument('--deterministic', action="store_true", help="deterministic mode")

    parser.add_argument('--q_transform', type=str, default='qadv', metavar='S', help="q_transform (default: qrelu)")

    parser.add_argument('--gradient', action="store_true", help="aug gradient")

    parser.add_argument('--policy_freq', type=int, default=1, metavar='N', help="policy_freq (default: 1)")

    parser.add_argument('--cut', type=float, default=1.0, metavar='G', help="cut (default: 1.0)")
    parser.add_argument('--times', type=int, default=1, metavar='N', help="times (default: 1)")

    parser.add_argument('--epsilon', type=float, default=0.0, metavar='G', help="eps greedy (default: 0.0)")
    parser.add_argument('--entropy_alpha', type=float, default=0.02, metavar='G', help="entropy_alpha (default: 0.02)")


    return parser.parse_args()


def evaluate(env, agent, steps):
    episodes = 10
    returns = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        state = env.reset()
        episode_reward = 0.
        done = False
        while not done:
            action = agent.sample_action(state, eval=True)[0]
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        returns[i] = episode_reward

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
          f'reward: {mean_return:<5.1f}  '
          f'std: {std_return:<5.1f}')
    print(returns)
    print('-' * 60)
    return mean_return


def main(args=None, logger=None, id=None):

    device = torch.device(args.cuda)

    dir = "record"
    # dir = "test"
    log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}', f'ratio={args.ratio}',
                           f'seed={args.seed}')
    writer = SummaryWriter(log_dir)

    # Initial environment
    env = gym.make(args.env_name)
    vec_enc = VectorEnv(args.env_name, 20, args.seed)

    eval_env = copy.deepcopy((env))
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    print(action_size)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    memory_size = 1e6
    num_steps = args.num_steps
    start_steps = 1500 # 20 samples per step, total 30000 samples for warm-up
    eval_interval = 15000
    updates_per_step = 1
    batch_size = args.batch_size
    log_interval = 10

    memory = ReplayMemory(state_size, action_size, memory_size, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, memory_size, device)

    agent = QVPO(args, state_size, env.action_space, memory, diffusion_memory, device)

    steps = 0
    best_result = -float('inf')

    states = vec_enc.reset()
    # 生成保存目录，添加当前日期和时间
    save_dir = os.path.join('./results', f"{prefix}_{name}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{id}_qvpo_s{args.seed}.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["wall_time", "step", "return"])

    # while steps < num_steps:
    for i in tqdm(range(num_steps), desc="Training Progress"):
        done = False

        if start_steps > steps:
            actions = []
            for j in range(20):
                actions.append(env.action_space.sample())
        else:
            actions = agent.sample_action(states, eval=False)
        
        srdi_list, next_states = vec_enc.step(actions)

        steps += 1
        for k, srdi in enumerate(srdi_list):
            next_state, reward, done, _ = srdi
            action = actions[k]
            state = states[k]
            mask = 0.0 if done else args.gamma
            agent.append_memory(state, action, reward, next_state, mask)

        states = next_states

        if steps >= start_steps:
            agent.train(steps, updates_per_step, batch_size=batch_size, log_writer=writer)
            agent.entropy_alpha = min(args.entropy_alpha, max(0.002, args.entropy_alpha-steps/num_steps*args.entropy_alpha))

        if steps % eval_interval == 0:
            tmp_result = evaluate(eval_env, agent, steps)

            if tmp_result > best_result:
                best_result = tmp_result
                agent.save_model(os.path.join('./results', prefix + '_' + name), id=id)
            
            # save to csv
            with open(os.path.join(save_dir, f"{id}_qvpo_s{args.seed}.csv"), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S"), steps, tmp_result])


if __name__ == "__main__":
    args = readParser()
    if args.target_sample == -1:
        args.target_sample = args.behavior_sample


    ## settings
    prefix = 'qvpo'
    name = args.env_name
    keys = ("epoch", "reward")
    times = args.times
    id = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    logger = Logger(name=name, keys=keys, max_epochs=int(args.num_steps)+2100, times=times, config=args, path=os.path.join('./results', prefix + '_' + name), id=id)


    ## run
    for time in range(times):
        main(args, logger=logger, id=id+"_"+str(time))

    logger.save(os.path.join('./results', prefix + '_' + name), id=id)
