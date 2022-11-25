import os
from os.path import join, dirname, abspath

import numpy as np
import torch
import pandas as pd


def save(X, y, optimizer_name, function_name):
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    results_array = torch.cat((X, y), dim=1).detach().numpy()
    results_cols = [f'X_i' for i in range(X.shape[1])]

    # retrieves the plotting data based on the name of this column - must be called y (or change plotting script)
    results_cols.append('y')
    
    result_path = join(join(join(dirname(dirname(abspath(__file__))), 'results', function_name), optimizer_name))
    run_index = os.listdir(result_path)
    os.makedirs(result_path, exist_ok=True)

    run_index = len(os.listdir(result_path))
    results_df = pd.DataFrame(results_array, columns=results_cols)
    print(f'Saving run at {result_path}...')
    results_df.to_csv(f"{result_path}/run_{run_index}.csv", index=False)
    print(f'Saved.')
    