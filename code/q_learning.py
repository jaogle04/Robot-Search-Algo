"""
Training and evaluating the q-learning agent on the LunarRoverEnv
"""

from lunar_rover_env import LunarRoverEnv
from collections import defaultdict
import numpy as np

"""
Turning Minigrid's observation into dict key
"""
def observation_to_state(observation):
    image = observation["image"]
    dir = int(observation["dir"])
    obs_key = (image.tobytes(), image.shape, dir)
    return obs_key

def choose_action(q_table, state, epsilon, action_count, rng):
    if rng.random() < epsilon:
        return int(rng.integers(action_count))
        
    return int(np.argmax(q_table[state]))

def train(episodes=8000, alpha=0.1, gamma=0.95, epislon_start=1.0,
          epsilon_end=0.05, seed=7):
    """
    When training, don't use the env random grid generator to generate a new
    grid each episode. Seed will make sure the grid map stays the same during
    the training process.
    """
    # declare the environment
    env = LunarRoverEnv(movement_failure_probability=0.15)

    # create q_table number of entries based off total number of possible actions
    # in the environment
    q_table = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float64))
    rng = np.random.default_rng(seed)
    success = []
    rewards = []
    
    for episode in range(episodes):
        # resetting/initializing environemnt for new training episode
        observation, _ = env.reset(seed=seed)
        state = observation_to_state(observation)
        eps_rew = 0.0
        terminated = False
        truncated = False

        frac = episode / max(episodes - 1, 1)
        epsilon = epislon_start + frac * (epsilon_end - epislon_start)

        while not (terminated or truncated):
            action = choose_action(q_table, state, epsilon, env.action_space.n, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state(next_obs)

            # Q-learning update
            next_val = 0.0 if terminated else np.max(q_table[next_state])
            td_target = reward + gamma * next_val
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
            eps_rew += reward

        rewards.append(eps_rew)
        success.append(int(terminated))

    env.close()
    return q_table

            

    return q_table

def evaluate(q_table, episodes=20, render=False, seed=7):
    pass

if __name__ == "__main__":
    q_table = train() 
    evaluate(q_table=q_table, render=True)
