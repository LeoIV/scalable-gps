import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    basedir = "../results"

    algo_benchs = {}

    # descend into directories and read function values from files
    for path, subdirs, files in os.walk(basedir):
        for name in files:
            if "run" in name:
                bench = path.split("/")[-2]
                algo = path.split("/")[-1]
                y = pd.read_csv(os.path.join(path, name)).to_numpy()[:, -1]
                y = np.array([max(y[:i]) for i in range(1, len(y) + 1)])
                if not bench in algo_benchs:
                    algo_benchs[bench] = {}
                if not algo in algo_benchs[bench]:
                    algo_benchs[bench][algo] = []
                algo_benchs[bench][algo].append(y)

    # create figure with as many columns as benchmarks
    fig, axs = plt.subplots(1, len(algo_benchs), figsize=(10, 5))

    # plot the benchmarks
    for i, bench in enumerate(algo_benchs.keys()):
        ax = axs[i] if len(algo_benchs) > 1 else axs
        for algo, y in algo_benchs[bench].items():
            # max length for this algorithm on this benchmark
            maxlen = max([len(yy) for yy in y])
            y = -np.array([np.concatenate([yy, np.ones(maxlen - len(yy)) * yy[-1]]) for yy in y])
            mean = np.mean(y, axis=0)
            # compute standard error of the mean
            std = np.std(y, ddof=1, axis=0) / np.sqrt(len(mean))
            ax.plot(np.arange(len(mean)), mean, label=algo, marker="x" if algo == "TurBO" else "^", markevery=50)
            ax.fill_between(np.arange(len(std)), mean - std, mean + std, alpha=0.5)
            ax.set_ylabel("Regret")
            ax.set_xlabel("Number of function evaluations")
        ax.set_title(bench)
        ax.legend()
        ax.set_yscale("log")
    plt.tight_layout()
    plt.show()
    fig.savefig("../data/results.png")