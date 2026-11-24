import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn import DQN, ReplayMemory
from lunar_rover_env import LunarRoverEnv


def observationToTensor(observation):
    """Convert a MiniGrid observation into a flat, normalized DQN input."""
    image = np.asarray(observation["image"], dtype=np.float32)
    direction = np.asarray([observation["direction"]], dtype=np.float32) / 3.0
    pixels = image.reshape(-1) / 10.0
    return torch.as_tensor(
        np.concatenate((pixels, direction)),
        dtype=torch.float32,
    )


def linearEpsilon(start, end, total_episodes, episode):
    if total_episodes <= 1:
        return end
    fraction = episode / (total_episodes - 1)
    return start + fraction * (end - start)


def train(
    episodes=500,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    learning_rate=1e-3,
    batch_size=128,
    memory_capacity=10000,
    target_update_frequency=100,
    seed=7,
):
    env = LunarRoverEnv(movement_failure_probability=0.15)
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    observation, _ = env.reset(seed=seed)
    initial_state = observationToTensor(observation)
    n_observations = initial_state.numel()
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions)
    target_net = DQN(n_observations, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_capacity)

    rewards = []
    successes = []
    steps = []
    optimization_steps = 0

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed)
        state = observationToTensor(observation)
        epsilon = linearEpsilon(
            epsilon_start,
            epsilon_end,
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
                    action = int(policy_net(state.unsqueeze(0)).argmax(dim=1).item())

            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = None if terminated else observationToTensor(next_observation)

            memory.push(state, action, next_state, reward)
            state = next_state
            episode_reward += reward
            episode_steps += 1

            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)

                batch_states = torch.stack([transition.state for transition in transitions])
                batch_actions = torch.tensor(
                    [transition.action for transition in transitions],
                    dtype=torch.long,
                ).unsqueeze(1)
                batch_rewards = torch.tensor(
                    [transition.reward for transition in transitions],
                    dtype=torch.float32,
                )

                non_terminal_mask = torch.tensor(
                    [transition.next_state is not None for transition in transitions],
                    dtype=torch.bool,
                )
                non_terminal_states = torch.stack(
                    [
                        transition.next_state
                        for transition in transitions
                        if transition.next_state is not None
                    ]
                )

                state_action_values = policy_net(batch_states).gather(1, batch_actions).squeeze(1)
                next_state_values = torch.zeros(batch_size)
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

    env.close()
    return policy_net, rewards, successes, steps


def evaluate(policy_net, episodes=20, render=False, seed=None):
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

    for _ in range(episodes):
        observation, _ = env.reset(seed=seed)
        state = observationToTensor(observation)
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
                    state = observationToTensor(next_observation)

        tot_rewards.append(round(episode_reward, 3))
        tot_successes.append(int(terminated))
        tot_steps.append(episode_steps)

    env.close()
    return tot_rewards, tot_successes, tot_steps


# Keep eval as the public loop name requested by the scaffold.
def eval(policy_net, episodes=20, render=False, seed=None):
    return evaluate(policy_net, episodes=episodes, render=render, seed=seed)


def logData(
    train_rewards=None,
    train_steps=None,
    train_success=None,
    eval_rewards=None,
    eval_steps=None,
    eval_success=None,
    output="./results/dqn.csv",
):
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if train_rewards is not None:
        with open(output, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["episodes", "Successful Run", "training rewards", "training steps"]
            )
            writer.writerows(
                zip(
                    range(len(train_rewards)),
                    train_success,
                    train_rewards,
                    train_steps,
                )
            )

    basename, extension = os.path.splitext(output)
    eval_output = f"{basename}_evaluation{extension}"
    with open(eval_output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["episodes", "Successful Run", "Evaluation Rewards", "Evaluation Steps"]
        )
        writer.writerows(
            zip(
                range(len(eval_rewards)),
                eval_success,
                eval_rewards,
                eval_steps,
            )
        )


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
    args = parser.parse_args()

    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error("episode arguments must be positive")
    if args.gamma < 0 or args.epsilon_start < 0 or args.epsilon_end < 0:
        parser.error("gamma and epsilon arguments must be positive")
    if args.batch_size < 1 or args.memory_capacity < 1:
        parser.error("batch size and memory capacity must be positive")

    policy_net, train_rewards, train_successes, train_steps = train(
        episodes=args.episodes,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        memory_capacity=args.memory_capacity,
        target_update_frequency=args.target_update_frequency,
        seed=args.seed,
    )

    eval_seed = None if args.random_seed == "Y" else args.seed
    eval_rewards, eval_successes, eval_steps = evaluate(
        policy_net,
        episodes=args.eval_episodes,
        render=args.render_eval == "Y",
        seed=eval_seed,
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
