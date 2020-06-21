import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
A2C_FILE = 'outputs/1592207112-a2c-70000/scores.json'
SMA2C_FILE = 'outputs/1592391802-sma2c-4000-smaller-mid'
OUTPUT = './res.png'
def main():  

    scores_a2c = json.load(open(A2C_FILE))
    scores_sma2c = json.load(open(SMA2C_FILE))
    # df_a2c = pd.read_json(A2C_FILE)
    # rolling = df_a2c.rolling(window=50).mean()
    # ax = rolling.plot.line()

    plt.title("Average score per episode (N=500)")

    # smooth = pd.rolling_mean(scores_a2c,windows=round(np.std(scores_a2c)))
    # plt.plot(smooth)
    scores_a2c = np.array([x[1] for x in scores_a2c])
    scores_sma2c = np.array([x[1] for x in scores_sma2c])
    # print(all_scores[:10])
    N = 500
    x = range(len(scores_a2c))
    plt.ylim((0, 30))
    plt.scatter(range(len(scores_a2c)), scores_a2c, marker='o', s=2, alpha=0.5, label='a2c')
    plt.scatter(range(len(scores_sma2c)), scores_sma2c, marker='o', s=2, alpha=0.5, label='sma2c')
    
    # plt.plot(np.convolve(scores_a2c, np.ones((N,))/N,
                            # mode='valid'),  label='A2C')
    z = np.polyfit(range(len(scores_a2c)), scores_a2c, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--')
    z2 = np.polyfit(range(len(scores_sma2c)), scores_sma2c, 1)
    p2 = np.poly1d(z2)
    plt.plot(x, p2(x), 'g--')
    # plt.plot(np.convolve(scores_sma2c, np.ones((N,))/N,
    #                         mode='valid'),  label='SMA2C')
    # z2 = np.polyfit(range(len(scores_sma2c)), scores_sma2c, 1)
    plt.legend()
    plt.savefig(OUTPUT)
    return
    

if __name__ == "__main__":
    main()
