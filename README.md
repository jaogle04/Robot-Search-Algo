# Lunar Robot Search Algorithm

## Setup

From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Q-learning

Train and evaluate Q-learning:

```bash
python code/q_learning_training.py \
  --episodes 1000 \
  --eval-episodes 100 \
  --epsilon-start 1.0 \
  --epsilon-end 0.05 \
  --alpha 0.1 \
  --gamma 0.95 \
  --seed 7 \
  --random-seed N \
  --render-eval N \
  --data-output ./results/run.csv \
  --model-output ./models/q_learn.pkl
```

Training data is written to `--data-output`. Evaluation data is written to the
same path with `_evaluation` inserted before the file extension (for example,
`./results/run_evaluation.csv`). The trained policy is written to
`--model-output`.

Run a random-policy baseline instead of training:

```bash
python code/q_learning_training.py \
  --random-policy Y \
  --episodes 1000 \
  --render-eval N \
  --data-output ./results/random_policy_run.csv
```

Common Q-learning options:

| Option | Default | Description |
| --- | --- | --- |
| `--episodes` | `1000` | Training episodes, or baseline episodes with `--random-policy Y` |
| `--eval-episodes` | `20` | Evaluation episodes |
| `--random-policy` | `N` | Set to `Y` to run the random-policy baseline |
| `--epsilon-start` | `1.0` | Initial exploration rate |
| `--epsilon-end` | `0.05` | Final exploration rate |
| `--alpha` | `0.1` | Learning rate |
| `--gamma` | `0.95` | Discount factor |
| `--seed` | `7` | Training/environment seed |
| `--random-seed` | `N` | Set to `Y` to generate a new evaluation map each episode |
| `--render-eval` | `True` | Set to `N` to disable evaluation rendering |
| `--data-output` | `./results/run.csv` | CSV output path |
| `--model-output` | `./models/q_learn.pkl` | Saved Q-table path |

## DQN

Train and evaluate DQN:

```bash
python code/dqn_training.py \
  --episodes 3000 \
  --eval-episodes 200 \
  --gamma 0.95 \
  --epsilon-start 1.0 \
  --epsilon-end 0.05 \
  --learning-rate 0.001 \
  --batch-size 128 \
  --memory-capacity 10000 \
  --target-update-frequency 100 \
  --seed 7 \
  --random-seed Y \
  --render-eval N \
  --output ./results/dqn.csv \
  --model-output ./models/dqn_policy.pt
```

Training data is written to `--output`, and evaluation data is written to the
same path with `_evaluation` inserted before the extension (for example,
`./results/dqn_evaluation.csv`). The trained network is saved to
`--model-output`.

Common DQN options:

| Option | Default | Description |
| --- | --- | --- |
| `--episodes` | `500` | Training episodes |
| `--eval-episodes` | `20` | Evaluation episodes |
| `--gamma` | `0.95` | Discount factor |
| `--epsilon-start` | `1.0` | Initial exploration rate |
| `--epsilon-end` | `0.05` | Final exploration rate |
| `--learning-rate` | `0.001` | Optimizer learning rate |
| `--batch-size` | `128` | Replay-batch size |
| `--memory-capacity` | `10000` | Replay-memory capacity |
| `--target-update-frequency` | `100` | Episodes between target-network updates |
| `--seed` | `7` | Training/environment seed |
| `--random-seed` | `N` | Set to `Y` to generate new evaluation maps |
| `--render-eval` | `N` | Set to `Y` to render evaluation episodes |
| `--output` | `./results/dqn.csv` | CSV output path |
| `--model-output` | `./models/dqn_policy.pt` | Saved DQN policy path |

The DQN wrapper script supports the same training flow:

```bash
./dqn_script run
./dqn_script analyze
./dqn_script test
```

## Test Saved Policies

Run 10 rendered evaluation episodes with a saved DQN policy:

```bash
python code/test_dqn.py --model ./models/dqn_policy.pt --episodes 10
```

Run 10 rendered evaluation episodes with a saved Q-table:

```bash
python code/test_qlearn.py --model ./models/q_learn.pkl --episodes 10
```

Both test scripts accept `--seed SEED` to repeat a specific environment layout;
without `--seed`, they generate independently varied maps.

## Analyze Results

For normal training results:

```bash
python code/analyze_results.py \
  --input-file ./results/run.csv \
  --output-file ./results/run \
  --window 100
```

This reads the matching `*_evaluation.csv` file and writes training and
evaluation plots.

For a random-policy CSV:

```bash
python code/analyze_results.py \
  --input-file ./results/random_policy_run_evaluation.csv \
  --rand-policy Y \
  --output-file ./results/random_policy_results \
  --window 100
```
