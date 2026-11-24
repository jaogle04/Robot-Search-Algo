import argparse
import os
import random
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from q_learning import epsilonDecay
from analyze_results import logData
from dqn import DQN, ReplayMemory
from lunar_rover_env import LunarRoverEnv


'''
@name observationToTensor 
@brief Converts the current observation dict inside minigrid to a tensor
for the pytorch library
'''
def observationToTensor(observation):
    """Convert a MiniGrid observation into a flat, normalized DQN input."""
    image = np.asarray(observation["image"], dtype=np.float32)
    direction = np.asarray([observation["direction"]], dtype=np.float32) / 3.0
    pixels = image.reshape(-1) / 10.0
    return torch.as_tensor(
        np.concatenate((pixels, direction)),
        dtype=torch.float32,
    )


'''
@name train
@brief Trains a DQN RL model to train the agent to travel through new grids
'''
def train(
    device,
    episodes=500,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    learning_rate=1e-3,
    batch_size=128,
    memory_capacity=10000,
    target_update_frequency=100,
    torch_seed=7,
    eps_seed=None,
):
    '''
    Initializing and preparing the environemnt
    '''
    env = LunarRoverEnv(movement_failure_probability=0.15)
    rng = random.Random(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)

    observation, _ = env.reset(seed=eps_seed)
    initial_state = observationToTensor(observation).to(device)
    n_observations = initial_state.numel()
    n_actions = env.action_space.n

    '''
    Setting up the DQN 
    '''
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_capacity)

    rewards = []
    successes = []
    steps = []
    optimization_steps = 0

    '''
    Optimization loop
    '''
    with tqdm(range(episodes), desc='Training', unit='episode', dynamic_ncols=True) as progress:
        for episode in progress:
            observation, _ = env.reset(seed=seed)
            state = observationToTensor(observation).to(device)
            epsilon = epsilonDecay(
                epsilon_end,
                epsilon_start,
                episodes,
                episode,
            )

            episode_reward = 0.0
            episode_steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                if rng.random() < epsilon:
                    action = rng.randrange(n_actions)
                else:
                    with torch.no_grad():
                        state = state.to(device)
                        action = int(policy_net(state.unsqueeze(0)).argmax(dim=1).item())

                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = (
                    None
                    if terminated or truncated
                    else observationToTensor(next_observation).to(device)
                )

                '''
                Storing the transition decision into memory for later
                '''
                memory.push(state, action, next_state, reward)
                state = next_state
                episode_reward += reward
                episode_steps += 1

                if len(memory) >= batch_size:
                    '''
                    Here we are grabbing a sample from our list of previous transitions 
                    '''
                    transitions = memory.sample(batch_size) 
                    batch_states = torch.stack(
                        [transition.state.to(device) for transition in transitions]
                    )
                    batch_actions = torch.tensor(
                        [transition.action for transition in transitions],
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(1)
                    batch_rewards = torch.tensor(
                        [transition.reward for transition in transitions],
                        dtype=torch.float32,
                        device=device,
                    )

                    non_terminal_mask = torch.tensor(
                        [transition.next_state is not None for transition in transitions],
                        dtype=torch.bool,
                        device=device,
                    )

                    non_terminal_transitions = [
                        transition
                        for transition in transitions
                        if transition.next_state is not None
                    ]

                    next_state_values = torch.zeros(batch_size, device=device)

                    if non_terminal_transitions:
                        non_terminal_states = torch.stack(
                            [
                                transition.next_state.to(device)
                                for transition in non_terminal_transitions
                            ]
                        )

                        with torch.no_grad():
                            next_state_values[non_terminal_mask] = target_net(
                                non_terminal_states
                            ).max(dim=1).values

                    state_action_values = policy_net(batch_states).gather(1, batch_actions).squeeze(1)
                    next_state_values = torch.zeros(batch_size, device=device)
                    if non_terminal_states.shape[0] > 0:
                        with torch.no_grad():
                            next_state_values[non_terminal_mask] = (
                                target_net(non_terminal_states).max(dim=1).values
                            )

                    expected_state_action_values = (
                        batch_rewards + gamma * next_state_values
                    )
                    loss = F.smooth_l1_loss(
                        state_action_values,
                        expected_state_action_values,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                    optimizer.step()

                    optimization_steps += 1
                    if optimization_steps % target_update_frequency == 0:
                        target_net.load_state_dict(policy_net.state_dict())

            rewards.append(round(episode_reward, 3))
            successes.append(int(terminated))
            steps.append(episode_steps)

            '''
            updating tqdm to track progress in train loop
            '''
            progress.set_postfix(
                reward=f"{episode_reward:.2f}",
                success=f"{sum(successes)}/{len(successes)}",
                epsilon=f"{epsilon:.3f}",
            )

    env.close()
    return policy_net, rewards, successes, steps


'''
@name evaluate
@brief evaluates the policy model created during train()
'''
def evaluate(device, policy_net, episodes=20, render=False, seed=None):
    if render:
        env = LunarRoverEnv(
            movement_failure_probability=0.15,
            render_mode="human",
        )
    else:
        env = LunarRoverEnv(movement_failure_probability=0.15)

    policy_net.eval()
    tot_rewards = []
    tot_successes = []
    tot_steps = []

    with tqdm(range(episodes), desc='Evaluation', unit='episode', dynamic_ncols=True) as progress:
        for episode in progress:
            observation, _ = env.reset(seed=seed)
            state = observationToTensor(observation).to(device)
            episode_reward = 0.0
            episode_steps = 0
            terminated = False
            truncated = False

            with torch.no_grad():
                while not (terminated or truncated):
                    action = int(
                        policy_net(state.unsqueeze(0)).argmax(dim=1).item()
                    )
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    episode_steps += 1
                    if not (terminated or truncated):
                        state = observationToTensor(next_observation).to(device)

            tot_rewards.append(round(episode_reward, 3))
            tot_successes.append(int(terminated))
            tot_steps.append(episode_steps)

            '''
            updating tqdm to track progress in train loop
            '''
            progress.set_postfix(
                reward=f"{episode_reward:.2f}",
                success=f"{sum(tot_successes)}/{len(tot_successes)}",
            )

    env.close()
    return tot_rewards, tot_successes, tot_steps


# Keep eval as the public loop name requested by the scaffold.
def eval(policy_net, episodes=20, render=False, seed=None):
    device = next(policy_net.parameters()).device
    return evaluate(device, policy_net, episodes=episodes, render=render, seed=seed)


'''
@name savePolicy
@brief saves and outputs the given policy produced from DQN training in a dict-style
format so files will be able to import and use it
'''
def savePolicy(policy_net, output):
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "state_dict": policy_net.state_dict(),
            "n_observations": policy_net.layer1.in_features,
            "n_actions": policy_net.layer3.out_features,
        },
        output,
    )


'''
@name loadPolicy
@brief Can be used to load in a saved policy in dict-style format to be used for testing
after training and evaluation
'''
def loadPolicy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy_net = DQN(
        checkpoint["n_observations"],
        checkpoint["n_actions"],
    ).to(device)
    policy_net.load_state_dict(checkpoint["state_dict"])
    policy_net.eval()
    return policy_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-capacity", type=int, default=10000)
    parser.add_argument("--target-update-frequency", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-seed", choices=("Y", "N"), default="N")
    parser.add_argument("--render-eval", choices=("Y", "N"), default="N")
    parser.add_argument("--output", type=str, default="./results/dqn.csv")
    parser.add_argument("--model-output", type=str, default="./models/dqn_policy.pt")
    args = parser.parse_args()

    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error("episode arguments must be positive")
    if args.gamma < 0 or args.epsilon_start < 0 or args.epsilon_end < 0:
        parser.error("gamma and epsilon arguments must be positive")
    if args.batch_size < 1 or args.memory_capacity < 1:
        parser.error("batch size and memory capacity must be positive")

    '''Determining if using cuda or not'''
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    seed = None if args.random_seed == "Y" else args.seed
    policy_net, train_rewards, train_successes, train_steps = train(
        device,
        episodes=args.episodes,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        memory_capacity=args.memory_capacity,
        target_update_frequency=args.target_update_frequency,
        torch_seed = args.seed,
        eps_seed=seed,
    )

    savePolicy(policy_net, args.model_output)

    eval_rewards, eval_successes, eval_steps = evaluate(
        device,
        policy_net,
        episodes=args.eval_episodes,
        render=args.render_eval == "Y",
        seed=seed,
    )
    logData(
        train_rewards,
        train_steps,
        train_successes,
        eval_rewards,
        eval_steps,
        eval_successes,
        args.output,
    )
