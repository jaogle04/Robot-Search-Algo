# Lunar Robot Search Algorithm
## Setup Environment
run:
```
pip -r install requirements.txt
```

## Execute Instructions
### Executing using given bash script
- shell script for running training and evaluating results
```bash
./run_script
```
- this will run the training produced results from the agent interacting in the environment as well as analyze the results
- options:
```
Usage: ./run_script {full_run|analyze|random_policy|analyze_policy}
```
- `./run_script full_run`: This will perform agent run for random policy (baseline comparison), default analyze with no random seed for evaulation, and then a run with random seed for evaluation. It will then analyze these results and create plot graphs for each of them in the `results` directory.
- `./run_script analyze`: This will analyze the results from a full run. This is useful if you have already performed a full run analysis and don't want run a new train process to gather for statistics from the results.
- `./run_script random_policy`: Runs a full baseline policy run, analyzes, and produces the plots from the run 
- `./run_script analyze_policy`: Analyze just the results from the random_policy/baseline comparison run


### Running Agent through Python Script
- If you want to execute through the `./code/run_agent.py` script:
```shell
python code/run_agent.py --episodes 1000 --eval-episodes 50 --epsilon-start 1.0 \
    --epsilon-end 0.05 --alpha 0.1 --gamma 0.95 --seed=7 \
    --render-eval N --random-seed N --output ./results/run.csv
```
>[!NOTE] The parameters shown here are the default parameters set in the `run_agent.py` script and would result in the same outcome if no input arguments were applied
```
usage: run_agent.py [-h] [--episodes EPISODES] [--eval-episodes EVAL_EPISODES] [--random-policy RANDOM_POLICY]
                    [--epsilon-start EPSILON_START] [--epsilon-end EPSILON_END] [--alpha ALPHA] [--gamma GAMMA]
                    [--seed SEED] [--random-seed RANDOM_SEED] [--render-eval RENDER_EVAL] [--output OUTPUT]
```

Arguments:
- `--episodes` is the total number of episodes used for training
- `--eval-episodes` total number of episodes used for evaluation (ignored when random policy is checked `Y`)
- `--random-policy` options `(Y/N)` and performs random policy evaluation. Inputs used for Q-learning will be ignored
- `--epsilon-start`
- `--epsilon-end`
- `--alpha` -- learning rate
- `--gamma` -- discount factor
- `--seed`  -- used for keeping grids between episodes
- `--random-seed` -- options `(Y/N)` and only effects evalution section, generates new grid during each episode
- `--render-eval` -- options `(Y/N)` and only effects evaluation, render the evalution episodes to see agent moving through grid
- `--output` -- where results are saved to, random policy and evaluation data add `_evaluation` at the end of the basename

### Analyzing Agent
- Analyze results using `./code/analyze_results.py` with input parameters:
```
usage: analyze_results.py [-h] [--input-file INPUT_FILE] [--rand-policy RAND_POLICY] [--output-file OUTPUT_FILE]
                          [--window WINDOW]
```
Arguments:
- `--input-file` csv resulting from training and evaluting the agent or just evaluting in the case of random policy
- `--rand-policy` is `(Y/N)` input telling the program to only run on dataset since random policy run only produces one csv unlike a Q-learning run
- `--output-file` will be the basename for the resulting plot pngs (do not need to include `.png` at the end)
- `--window` is an integer value used for moving average that cleans up the training data plot
