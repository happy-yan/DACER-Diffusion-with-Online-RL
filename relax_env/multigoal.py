import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box


class MultiGoalEnv(Env):
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """

    def __init__(self, goal_reward=10, actuation_cost_coeff=30,
                 distance_cost_coeff=1, init_sigma=0.1):
        super().__init__()

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            [
                [5, 0],
                [-5, 0],
                [0, 5],
                [0, -5]
            ],
            dtype=np.float32
        )
        self.goal_threshold = 1.
        self.goal_reward = goal_reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self.observation = None

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': []}
        self.spec = None

        self._ax = None
        self._env_lines = []
        self.fixed_plots = None
        self.dynamic_plots = []

    def reset(self, *, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        unclipped_observation = self.init_mu + self.init_sigma * \
                                self.np_random.normal(size=self.dynamics.s_dim)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        self.observation = np.clip(unclipped_observation, o_lb, o_ub).astype(np.float32)
        return self.observation, {}

    @property
    def observation_space(self):
        return Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            shape=None,
            dtype=np.float32
        )

    @property
    def action_space(self):
        return Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,),
            dtype=np.float32
        )

    def get_current_obs(self):
        return np.copy(self.observation)

    def step(self, action):
        action = action.ravel()

        a_lb, a_ub = self.action_space.low, self.action_space.high
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.dynamics.forward(self.observation, action, self.np_random)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        next_obs = np.clip(next_obs, o_lb, o_ub)

        self.observation = np.copy(next_obs)

        reward = self.compute_reward(self.observation, action)
        cur_position = self.observation
        dist_to_goal = np.amin([
            np.linalg.norm(cur_position - goal_position)
            for goal_position in self.goal_positions
        ])
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        return next_obs.astype(np.float32), reward, done, False, {'pos': next_obs}

    def plot(self, state, action):
        import matplotlib.pyplot as plt
        fig_env = plt.figure(figsize=(7, 7))
        self._ax = fig_env.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = []
        self._ax.set_xlim((-7, 7))
        self._ax.set_ylim((-7, 7))

        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')

        self._plot_position_cost(self._ax, state, action)

    def render(self, paths):
        import matplotlib.pyplot as plt
        if self._ax is None:
            self._init_plot()

        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []

        for path in paths:
            positions = np.stack([info['pos'] for info in path['env_infos']])
            xx = positions[:, 0]
            yy = positions[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b')

        plt.draw()
        plt.pause(0.01)

    def compute_reward(self, observation, action):
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = observation
        # noinspection PyTypeChecker
        goal_cost = self.distance_cost_coeff * np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward

    def _plot_position_cost(self, ax, state, action):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        
        sigma = 1.7
        goal_costs = np.sum([
            40 / (2 * np.pi * (sigma ** 2)) * np.exp(-((X - goal_x) ** 2 + (Y - goal_y) ** 2) / (2 * sigma ** 2))
            for goal_x, goal_y in self.goal_positions
        ], axis=0)

        # reward = np.clip(goal_costs, 0.2, 2.5)
        costs = goal_costs
        levels = np.linspace(np.min(costs), np.max(costs), 20)

        contours = ax.contour(X, Y, costs, levels=levels)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ro')

        X = state[:, 0]
        Y = state[:, 1]
        U = action[:, 0] * 3
        V = action[:, 1] * 3

        ax.quiver(X, Y, U, V, color="r", angles='xy',
                  scale_units='xy', scale=2, width=.005)

        return [contours, goal]

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def log_diagnostics(self, paths):
        n_goal = len(self.goal_positions)
        goal_reached = [False] * n_goal

        for path in paths:
            last_obs = path["observations"][-1]
            for i, goal in enumerate(self.goal_positions):
                if np.linalg.norm(last_obs - goal) < self.goal_threshold:
                    goal_reached[i] = True

        # logger.record_tabular('env:goals_reached', goal_reached.count(True))

    def horizon(self):
        return None


class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """

    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action, np_random):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
                     np_random.normal(size=self.s_dim)
        return state_next
