import matplotlib.pyplot as plt
import argparse
import os
import csv

from numpy import dtype

def plotData():
    pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input-file', type=str, default="./results/run.csv")
    args = p.parse_args()

    train_file = args.input_file
    bn, type = os.path.splitext(train_file)
    eval_file = f"{bn}_evaluation{type}"

    # grabbing train data for plotting
    with open(train_file, 'r') as f:
        pass

    # grabbing eval data
    with open(eval_file, 'r') as f:
        pass

    plotData()
