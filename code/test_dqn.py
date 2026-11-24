'''Run a saved DQN policy in a small number of rendered evaluation episodes.'''

import argparse

from dqn_training import evaluate, loadPolicy

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='./models/dqn_policy.pt')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional seed; leave unset to test on independently generated maps.',
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error('--episodes must be positive')

    policy_net = loadPolicy(args.model)
    rewards, successes, steps = evaluate(
        policy_net,
        episodes=args.episodes,
        render=True,
        seed=args.seed,
    )

    print(f'Episodes: {args.episodes}')
    print(f'Successes: {sum(successes)}/{args.episodes}')
    print(f'Average reward: {sum(rewards) / args.episodes:.3f}')
    print(f'Average steps: {sum(steps) / args.episodes:.1f}')


if __name__ == '__main__':
    main()
