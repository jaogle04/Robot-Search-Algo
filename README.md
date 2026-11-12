# Lunar Robot Search Algorithm
## Setup Environment
run:
```
pip -r install requirements.txt
```

## Execute Instructions
- shell script for running training and evaluating results
```bash
./run_script
```
- this will run the training produced results from the agent interacting in the environment as well as analyze the results

- If you want to execute through the python script:
```shell
python code/run_agent.py --episodes 1000 --eval-episodes 50 --epsilon-start 1.0 \
    --epsilon-end 0.05 --alpha 0.1 --gamma 0.95 --seed=7 \
    --render-eval N --random-seed N --output ./results/run.csv
```
>[!NOTE] The parameters shown here are the default parameters set in the python script and would result in the same outcome if no input arguments were applied
