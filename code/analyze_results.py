import matplotlib.pyplot as plt
import argparse
import os
import csv

def plotData(eps:list, rewards:list, steps:list, output:str):
    plt.plot(eps, rewards, '-', 'red')
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.title('Training Results')
    plt.savefig(f'{output}.png')
    
if __name__ == "__main__":
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
    args = p.parse_args()

    train_file = args.input_file
    bn, type = os.path.splitext(train_file)
    eval_file = f"{bn}_evaluation{type}"

    # grabbing train data for plotting
    with open(train_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            train_eps.append(row[0])
            train_success.append(row[1])
            train_rewards.append(row[2])
            train_steps.append(row[3])

    # grabbing eval data
    with open(eval_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            eval_eps.append(row[0])
            eval_success.append(row[1])
            eval_rewards.append(row[2])
            eval_steps.append(row[3])

    
    train_out = f'{args.output_file}_train'
    eval_out  = f'{args.output_file}_eval'
        

    plotData(train_eps, train_rewards, train_steps, train_out)
    plotData(eval_eps, eval_rewards, eval_steps, eval_out)
