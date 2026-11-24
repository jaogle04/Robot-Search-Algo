from q_learning import createQTable
import pickle
import argparse
from q_learning_training import evaluate_q_learning

def load_q_table(path):
    with open(path, "rb") as file:
        saved = pickle.load(file)  # Only load files you trust.

    q_table = createQTable(saved["num_actions"])
    q_table.update(saved["q_table"])
    return q_table

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='./models/q_learn.pkl')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional seed; leave unset to test on independently generated maps.',
    )
    args = parser.parse_args()

    q_table = load_q_table(args.model)

    rewards, successes, steps = evaluate_q_learning(q_table,
                                                 args.episodes,
                                                 render=True,
                                                 seed=args.seed)

    print(f'Episodes: {args.episodes}')
    print(f'Successes: {sum(successes)}/{args.episodes}')
    print(f'Average reward: {sum(rewards) / args.episodes:.3f}')
    print(f'Average steps: {sum(steps) / args.episodes:.1f}')

if __name__ == "__main__":
    main()
