import os
from copy import copy
from glob import glob
from os.path import join
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

from plotconfig import (
    COLORS,
    REGRETS,
    INIT_NBRS,
    NAMES,
    PLOT_LAYOUT,
)

plt.rcParams['font.family'] = 'serif'


# Some constants for better aesthetics
# '#377eb8', '#ff7f00', '#4daf4a'
# '#f781bf', '#a65628', '#984ea3'
# '#999999', '#e41a1c', '#dede00'


def process_funcs_args_kwargs(input_tuple):
    '''
    helper function for preprocessing to assure that the format of (func, args, kwargs is correct)
    '''
    if len(input_tuple) != 3:
        raise ValueError(
            f'Expected 3 elements (callable, list, dict), got {len(input_tuple)}')

    if not callable(input_tuple[0]):
        raise ValueError('Preprocessing function is not callable.')

    if type(input_tuple[1]) is not list:
        raise ValueError(
            'Second argument to preprocessing function is not a list.')

    if type(input_tuple[2]) is not dict:
        raise ValueError(
            'Third argument to preprocessing function is not a dict.')

    return input_tuple


def filter_paths(all_paths, included_names=None):
    all_names = [benchmark_path.split('/')[-1]
                 for benchmark_path in all_paths]
    if included_names is not None:
        used_paths = []
        used_names = []

        for path, name in zip(all_paths, all_names):
            if name in included_names:
                used_paths.append(path)
                used_names.append(name)
        return used_paths, used_names

    return all_paths, all_names


def get_files_from_experiment(experiment_name, benchmarks=None, acquisitions=None):
    '''
    For a specific expefiment, gets a dictionary of all the {benchmark: {method: [output_file_paths]}}
    as a dict, includiong all benchmarks and acquisition functions unless specified otherwise in 
    the arguments.
    '''
    paths_dict = {}
    all_benchmark_paths = glob(join(experiment_name, '*'))
    print(join(experiment_name, '*'))
    filtered_benchmark_paths, filtered_benchmark_names = filter_paths(
        all_benchmark_paths, benchmarks)

    # *ensures hidden files are not included
    for benchmark_path, benchmark_name in zip(filtered_benchmark_paths, filtered_benchmark_names):
        paths_dict[benchmark_name] = {}
        all_acq_paths = glob(join(benchmark_path, '*'))
        filtered_acq_paths, filtered_acq_names = filter_paths(
            all_acq_paths, acquisitions)

        for acq_path, acq_name in zip(filtered_acq_paths, filtered_acq_names):
            run_paths = glob(join(acq_path, '*'))
            paths_dict[benchmark_name][acq_name] = run_paths

    return paths_dict


def get_dataframe(paths, funcs_args_kwargs=None, idx=0):
    '''
    For a given benchmark and acquisition function (i.e. the relevant list of paths), 
    creates the dataframe that includes the relevant metrics.

    Parameters:
        paths: The paths to the experiments that should be included in the dataframe
        funcs_args_kwargs: List of tuples of preprocessing arguments,
    '''
    # ensure we grab the name from the right spot in the file structure
    names = [path.split('/')[-1].split('.')[0] for path in paths]

    # just create the dataframe and set the column names
    complete_df = pd.DataFrame(columns=names)

    # tracks the maximum possible length of the dataframe
    max_length = None

    for path, name in zip(paths, names):
        per_run_df = pd.read_csv(path)
        # this is where we get either the predictions or the true values
        if funcs_args_kwargs is not None:
            for func_arg_kwarg in funcs_args_kwargs:
                func, args, kwargs = process_funcs_args_kwargs(func_arg_kwarg)
                per_run_df = func(per_run_df, name, *args, **kwargs)

        complete_df.loc[:, name] = per_run_df.iloc[:, 0]
    return complete_df


def get_min(df, run_name, metric, minimize=True):
    min_observed = np.inf
    mins = np.zeros(len(df))

    for r, row in enumerate(df[metric]):
        if minimize:
            if row < min_observed:
                min_observed = row
            mins[r] = min_observed
        else:
            if -row < min_observed:
                min_observed = -row
            mins[r] = min_observed
    return pd.DataFrame(mins, columns=[run_name])


def compute_regret(df, run_name, regret, log=True):
    if log:
        mins = df.iloc[:, 0].apply(lambda x: np.log10(x + regret))
    else:
        mins = df.iloc[:, 0].apply(lambda x: x + regret)

    return pd.DataFrame(mins)


def plot_optimization(data_dict, preprocessing=None, title='benchmark', xlabel='X', ylabel='Y', fix_range=None,
                      only_plot=-1, names=None, predictions=False, init=2, n_markers=20, n_std=1, show_ylabel=True,
                      maxlen=0, plot_ax=None, first=True, show_noise=None):

    if plot_ax is None:
        fig, ax = plt.subplots(figsize=(25, 16))
    else:
        ax = plot_ax

    min_ = np.inf
    for run_name, files in data_dict.items():
        plot_layout = copy(PLOT_LAYOUT)
        plot_layout['c'] = COLORS.get(run_name, 'k')
        plot_layout['label'] = NAMES.get(run_name, 'Nameless Run')
        if plot_layout['label'] == 'Nameless Run':
            continue
        plot_layout['marker'] = '*'
        plot_layout['markersize'] = 10
        plot_layout['markeredgewidth'] = 2
        # preprocess the data for the set of runs
        result_dataframe = get_dataframe(files, preprocessing)
        # convert to array and plot
        data_array = result_dataframe.to_numpy()
        markevery = np.floor(maxlen / n_markers).astype(int)
        plot_layout['markevery'] = markevery
        if only_plot > 0:
            data_array = data_array[:, 0:only_plot]

        y_mean = data_array.mean(axis=1)
        y_std = sem(data_array, axis=1)
        X = np.arange(1, maxlen + 1)
        if fix_range is not None:
            ax.set_ylim(fix_range)

        ax.plot(X, y_mean, **plot_layout)
        ax.fill_between(X, y_mean - n_std * y_std, y_mean + n_std *
                        y_std, alpha=0.1, color=plot_layout['c'])
        ax.plot(X, y_mean - n_std * y_std, alpha=0.5, color=plot_layout['c'])
        ax.plot(X, y_mean + n_std * y_std, alpha=0.5, color=plot_layout['c'])
        min_ = min((y_mean - n_std * y_std).min(), min_)

    ax.axvline(x=init, color='k', linestyle=':', linewidth=4)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_title(title, fontsize=30)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=24)

    if first:
        handles, labels = ax.get_legend_handles_labels()
        sorted_indices = np.argsort(labels[:-1])
        sorted_indices = np.append(sorted_indices, len(labels) - 1)
        ax.legend(np.array(handles)[sorted_indices], np.array(
            labels)[sorted_indices], fontsize=24)
    if plot_ax is not None:
        return ax


def plot(experiment_name: str, algos: List[str], functions: List[str]):
    files = get_files_from_experiment(
        f'{os.getcwd()}/experiment_name', functions, algos)

    num_benchmarks = len(files)
    if num_benchmarks == 0:
        raise ValueError('No files')

    fig, ax = plt.subplots(1, num_benchmarks, figsize=(25, 9))
    for benchmark_idx, (benchmark_name, paths) in enumerate(files.items()):
        preprocessing = [(get_min, [], {'metric': 'y'}), (compute_regret, [], {
            'log': True, 'regret': REGRETS[benchmark_name]})]
        plot_optimization(paths,
                          xlabel='Iteration',
                          ylabel='Log Regret',
                          n_std=2,
                          preprocessing=preprocessing,
                          maxlen=150,
                          plot_ax=ax[benchmark_idx],
                          first=benchmark_idx == 0,
                          n_markers=10,
                          init=INIT_NBRS[benchmark_name],
                          title=benchmark_name,
                          show_ylabel=False,
                          )
    plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/results/{experiment_name}_{'_'.join(functions)}.pdf")
    plt.show()
