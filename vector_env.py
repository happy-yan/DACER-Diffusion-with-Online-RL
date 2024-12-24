import gym
import numpy as np

# 功能：接收一个环境名称，例如HalfCheetah-v3，以及环境个数，实现多个环境一起运行
# 每次step，输入的是一个batch的动作

class VectorEnv:
    def __init__(self, env_name, num_envs, seed):
        self.envs = [gym.make(env_name) for _ in range(num_envs)]
        self.num_envs = num_envs
        for i in range(num_envs):
            self.envs[i].seed(seed + i)
        
    def reset(self):
        return np.array([env.reset() for env in self.envs]) # [num_envs, state_size]
    
    def step(self, actions):
        # actions: [num_envs, action_size]
        # return: return_list, next_states
        srdi_list = []
        next_states = []
        for i, env in enumerate(self.envs):
            next_state, reward, done, _ = env.step(actions[i])
            srdi_list.append([next_state, reward, done, _])
            if done:
                next_state = env.reset()
            next_states.append(next_state)

        return srdi_list, np.array(next_states) # [num_envs, state_size]