import gymnasium

gymnasium.register(
    "MultiGoal-v0",
    entry_point="relax_env.multigoal:MultiGoalEnv",
    max_episode_steps=100,
)