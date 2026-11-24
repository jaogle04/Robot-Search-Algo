import matplotlib.pyplot as plt
import argparse
import os
import csv
import numpy as np
import math
from itertools import zip_longest

'''
Log info from Training
- writing data out to csv
'''
def logData(train_rewards:list | None=None, train_steps:list | None=None,
            train_success:list | None=None, eval_rewards:list | None=None,
            eval_steps:list | None=None, eval_success:list | None=None, output:str | None=None):

    # creating directory to store results in-case it does not exist
    dir = os.path.dirname(output)
    os.makedirs(dir, exist_ok=True) # skips creation if already exist

    # No training was performed on data being logged
    if train_rewards is not None:
        print('Saving Training results...')
        header = ['episodes',
                  'Successful Run',
                  'training rewards',
                  'training steps',]

        episodes = list(range(len(train_rewards))) # creating a episodes list with vals 0 -- len(train_rewards)-1
        results = zip_longest(episodes, train_success, train_rewards, train_steps)

        # write out training results
        with open(output, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in results:
                writer.writerow(row)

    bn, type = os.path.splitext(output)
    output_eval = f'{bn}_evaluation{type}'

    # write out eval results
    header = ['episodes',
              'Successful Run',
              'Evaluation Rewards',
              'Evaluation Steps']
            
    episodes = list(range(len(eval_rewards))) # creating a episodes list with vals 0 -- len(train_rewards)-1
    results = zip_longest(episodes, eval_success, eval_rewards, eval_steps)

    with open(output_eval, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)

    return

"""
Smoooths the results for training to make graph more readable
"""
def movingAvg(values, window):
    if len(values) == 0:
        return values

    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode='valid')

def plotData(episodes, rewards, steps, window, output, fig_title):
    rewards = movingAvg(rewards, window)
    steps = movingAvg(steps, window)
    episodes = episodes[window - 1 :]
    plt.figure(figsize=(5, 3))
    fig, axes = plt.subplots(2, 1, figsize=(10,8))
    axes[0].plot(episodes,
                 rewards,
                 color='tab:blue',
                 linewidth=0.8,
                 label='reward')
    axes[0].set_title('Rewards per Episodes')
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Rewards')
    axes[0].grid(True)

    axes[1].plot(episodes,
                 steps,
                 color='tab:blue',
                 linewidth=0.8,
                 label='reward')
    axes[1].set_title('Steps per Episodes')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Steps')
    axes[1].grid(True)

    fig.suptitle(f'{fig_title} Results')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output}.png', dpi=300, bbox_inches='tight')
    plt.close()    

def grabData(filename):
    epi, success, rewards, steps = [], [], [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            epi.append(int(row[0]))
            success.append(int(row[1]))
            rewards.append(float(row[2]))
            steps.append(float(row[3]))

    epi = np.asarray(epi)
    success = np.asarray(success)
    rewards = np.asarray(rewards)
    steps = np.asarray(steps)
    return epi, success, rewards, steps

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-file', type=str, default='./results/run.csv')
    p.add_argument('--rand-policy',
                   help='Mark (Y/N) if data being analyzed is from random policy run',
                   type=str,
                   default='N')
    p.add_argument('--output-file', type=str, default='./results/run')
    p.add_argument('--window', type=int, default=100)
    args = p.parse_args()

    if args.rand_policy == 'Y':
        rand_policy = True
    else:
        rand_policy = False

    print(f'Random Policy Analyzing results: {args.rand_policy}')
    if not rand_policy:
        print("not random policy")
        train_file = args.input_file
        bn, type = os.path.splitext(train_file)
        eval_file = f'{bn}_evaluation{type}'
        # grabbing train data for plotting
        train_eps, train_success, train_rewards, train_steps = grabData(train_file)
        # grabbing eval data
        eval_eps, eval_success, eval_rewards, eval_steps = grabData(eval_file)
    
        train_out = f'{args.output_file}_train'
        eval_out  = f'{args.output_file}_eval'
        
        plotData(train_eps, train_rewards, train_steps, args.window,train_out, fig_title='Training')

        #eval_window = min(args.window, len(eval_rewards))
        eval_window = math.ceil(len(eval_rewards)/15)
        eval_window = 30
        plotData(eval_eps, eval_rewards, eval_steps, eval_window, eval_out, fig_title='Evaluation')
    else:
        file = args.input_file
        eps, success, rewards, steps = grabData(file)
        out_file = f'{args.output_file}_eval'
        plotData(eps, rewards, steps, args.window, out_file, fig_title='Random Policy')
