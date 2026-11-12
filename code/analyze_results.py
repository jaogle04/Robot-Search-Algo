import matplotlib.pyplot as plt
import argparse
import os
import csv
import numpy as np
import math

"""
Smoooths the results for training to make graph more readable
"""
def movingAvg(values, window):
    if len(values) == 0:
        return values

    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode='valid')

def plotData(episodes, rewards, steps, window, output, type_data):
    if type_data == 'train':
        # smoothing training data
        fig_title = 'Training'
    else:
        fig_title = 'Evaluation'

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


if __name__ == '__main__':
    train_eps       = []
    train_rewards   = []
    train_steps     = []
    train_success   = []
    eval_eps        = []
    eval_rewards    = []
    eval_steps      = []
    eval_success    = []

    p = argparse.ArgumentParser()
    p.add_argument('--input-file', type=str, default='./results/run.csv')
    p.add_argument('--output-file', type=str, default='./results/run')
    p.add_argument('--window', type=int, default=100)
    args = p.parse_args()

    train_file = args.input_file
    bn, type = os.path.splitext(train_file)
    eval_file = f'{bn}_evaluation{type}'

    # grabbing train data for plotting
    with open(train_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            train_eps.append(int(row[0]))
            train_success.append(int(row[1]))
            train_rewards.append(float(row[2]))
            train_steps.append(float(row[3]))

    train_eps = np.asarray(train_eps)
    train_success = np.asarray(train_success)
    train_rewards = np.asarray(train_rewards)
    train_steps = np.asarray(train_steps)

    # grabbing eval data
    with open(eval_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            eval_eps.append(int(row[0]))
            eval_success.append(int(row[1]))
            eval_rewards.append(float(row[2]))
            eval_steps.append(float(row[3]))

    eval_eps = np.asarray(eval_eps)
    eval_success = np.asarray(eval_success)
    eval_rewards = np.asarray(eval_rewards)
    eval_steps = np.asarray(eval_steps)
    
    train_out = f'{args.output_file}_train'
    eval_out  = f'{args.output_file}_eval'
        

    plotData(train_eps, train_rewards, train_steps, args.window,train_out, type_data="train")

    #eval_window = min(args.window, len(eval_rewards))
    eval_window = math.ceil(len(eval_rewards)/15)
    eval_window = 30
    plotData(eval_eps, eval_rewards, eval_steps, eval_window, eval_out, type_data="eval")
