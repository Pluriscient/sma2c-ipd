import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--histogram")
    parser.add_argument("--output", default='scores.png')
    parser.add_argument("--combine", action='store_true')
    parser.add_argument("--title", default="Scores")
    parser.add_argument("score_file")
    # parser.add_argument("")
    args = parser.parse_args()

    scores = json.load(open(args.score_file))

    if args.combine:
        plt.title("Score per episode: A2C")
        all_scores = np.array([x[1] for x in scores])
        # print(all_scores[:10])
        N = 3000
        plt.plot(np.convolve(all_scores, np.ones((N,))/N,
                             mode='valid'),  label='score')
        z = np.polyfit(range(len(scores)), all_scores, 1)
        plt.ylim((-30, 60))
        plt.legend()
        plt.savefig(args.output)
        return
    scores = sorted(scores, key=lambda x: x[0])
    its = itertools.groupby(scores, lambda x: x[0])
    plt.title(args.title)
    for key, subit in its:
        print(key)
        changes = (list([x[1] for x in subit]))
        x = range(len(changes))
        plt.plot(changes, 'o',  label=key, markersize=2)
        z = np.polyfit(x, changes, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), 'r--')
    plt.legend()
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
