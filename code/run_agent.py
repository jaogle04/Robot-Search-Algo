from lunar_rover_env import LunarRoverEnv
from q_learning import *

"""
Log info from Training
- writing data out to csv
"""
def logData(train_rewards:list, train_steps:list, train_success:list,
            eval_rewards:list, eval_steps:list, eval_success:list, output:str):

    # creating directory to store results in-case it does not exist
    dir = os.path.dirname(output)
    os.makedirs(dir, exist_ok=True) # skips creation if already exist

    header = ['episodes',
              'Successful Run',
              'training rewards',
              'training steps',]

    episodes = list(range(len(train_rewards))) # creating a episodes list with vals 0 -- len(train_rewards)-1
    results = zip_longest(episodes, train_success, train_rewards, train_steps)

    # write out training results
    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)

    bn, type = os.path.splitext(output)
    output_eval = f"{bn}_evaluation{type}"

    # write out eval results
    header = ['episodes',
              'Successful Run',
              'Evaluation Rewards',
              'Evaluation Steps']
            
    episodes = list(range(len(eval_rewards))) # creating a episodes list with vals 0 -- len(train_rewards)-1
    results = zip_longest(episodes, eval_success, eval_rewards, eval_steps)

    with open(output_eval, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)

    return



def train(episodes=500, alpha=0.1, gamma=0.95, epislon_start=1.0,
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
    q_table = createQTable(env.action_space.n)
    rng = np.random.default_rng(seed)
    success = []
    rewards = []
    tot_steps = []
    
    for episode in range(episodes):
        # resetting/initializing environemnt for new training episode
        observation, _ = env.reset(seed=seed)
        state = observationToState(observation)
        eps_rew = 0.0
        steps = 0
        terminated = False
        truncated = False

        epsilon = epsilonDecay(epsilon_end, epislon_start, episodes,episode)

        while not (terminated or truncated):
            action = chooseAction(q_table, state, epsilon, env.action_space.n, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observationToState(next_obs)

            # Q-learning update
            q_table[state][action] += updateQTable(terminated, q_table, state,
                                                   next_state, action, reward, gamma, alpha)

            state = next_state
            eps_rew += reward
            steps += 1

        rewards.append(round(eps_rew, 3))
        success.append(int(terminated))
        tot_steps.append(steps)

    env.close()
    return q_table, rewards, success, tot_steps

def evaluate(q_table, episodes=20, render=False, seed=None):
    if render:
        env = LunarRoverEnv(movement_failure_probability=0.15, render_mode="human")
    else:
        env = LunarRoverEnv(movement_failure_probability=0.15)

    if seed == None:
        rng = np.random.default_rng()
    else: 
        rng = np.random.default_rng(seed)

    epsilon = 0
    tot_rewards = []
    tot_successes = []
    tot_steps = []

    for episode in range(episodes):
        steps = 0
        terminated = False
        truncated = False
        # not using seed here to see how it performs
        # across different grids
        observation, _ = env.reset(seed=seed) 
        state = observationToState(observation)
        eps_rew = 0

        while not (terminated or  truncated):
            # next action
            action = chooseAction(q_table, state, 0, env.action_space.n, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observationToState(next_obs)
            eps_rew += reward
            state = next_state
            steps += 1

        tot_rewards.append(round(eps_rew, 3))
        tot_steps.append(steps)
        tot_successes.append(int(terminated))

    env.close()
    return tot_rewards, tot_successes,tot_steps

if __name__ == "__main__":
    # Allowed inputs for user to allow different hyperparameters for training
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--epsilon-start', type=float, default=1.0)
    p.add_argument('--epsilon-end', type=float, default=0.05)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--random-seed',
                   help='random grid generation for each episode run on evaluation step (Y/N)',
                   type=str, default='N')
    p.add_argument('--render-eval', type=str, default=True)
    p.add_argument('--output', type=str, default='./results/run.csv') # output directory for data
    
    args = p.parse_args()

    # Check argument inputs
    if min(args.episodes, args.eval_episodes) < 1:
        p.error('episode arguments must positive')
    if min(args.epsilon_start, args.epsilon_end, args.alpha, args.gamma) < 0:
        p.error('epsilon and gamma arguments must be positive')
    if args.random_seed in ('Y'):
        eval_seed = None
    else: eval_seed = args.seed
    if args.render_eval in ('False', 'No', 'F', 'N'):
        rend = False
    else: rend = True 


    # create q_table implementing q-learning
    q_table, train_rewards, train_successes, train_steps = train(args.episodes, args.alpha,
                                                                 args.gamma, args.epsilon_start,
                                                                 args.epsilon_end, args.seed) 

    # evaluated the resulting table
    eval_rewards, eval_successes, eval_steps = evaluate(q_table, args.eval_episodes,
                                                        render=rend, seed=eval_seed)
    logData(train_rewards, train_steps, train_successes, eval_rewards, eval_steps, eval_successes, args.output)
