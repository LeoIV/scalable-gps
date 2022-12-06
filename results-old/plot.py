import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':

    basedir = "."


    algo_benchs = {}

    for path, subdirs, files in os.walk(basedir):
        for name in files:
            if "run" in name:
                bench = path.split("/")[-2]
                algo = path.split("/")[-1]
                y = pd.read_csv(os.path.join(path, name)).to_numpy()[:, -1]
                y = np.array([max(y[:i]) for i in range(1,len(y)+1)])
                if not bench in algo_benchs:
                    algo_benchs[bench] = {}
                if not algo in algo_benchs[bench]:
                    algo_benchs[bench][algo] = []
                algo_benchs[bench][algo].append(y)

    fig, axs = plt.subplots(1, len(algo_benchs), )

    for i, bench in enumerate(algo_benchs.keys()):
        ax = axs[i] if len(algo_benchs)>1 else axs
        for algo, y in algo_benchs[bench].items():
            maxlen = max([len(yy) for yy in y])
            y = [np.concatenate([yy, np.ones(maxlen-len(yy))*yy[-1]]) for yy in y]
            #y = [yy[:minlen] for yy in y]
            mean = np.mean(np.array(y), axis=0)
            std = np.std(y, ddof=1, axis=0) / np.sqrt(len(mean))
            ax.plot(np.arange(len(mean)), mean, label=algo)
            ax.fill_between(np.arange(len(std)), mean-std, mean+std, alpha=0.5)
        ax.set_title(bench)
        ax.legend()

    plt.show()