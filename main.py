import os

import numpy as np
from botorch.models import SingleTaskGP

from scalable_gps.objective import Ackley, Griewank
from scalable_gps.optimizer import ScalableOptimizer

if __name__ == '__main__':

    SAVE_PATH = os.path.join(os.getcwd(), "results")

    objectives = [Ackley(4), Griewank(10)]

    for objective in objectives:
        for use_dkl in [True, False]:
            for rep in range(3):
                print(f"Repetition {rep}")
                name = "TurBO-DKL" if use_dkl else "TurBO"
                turbo_kwargs = {
                    'model': SingleTaskGP,
                }
                seed = np.random.randint(1000)
                so = ScalableOptimizer(objective, num_parallel=4, turbo_kwargs={}, name=name, use_dkl=use_dkl,
                                       outer_iterations=5, save_path=SAVE_PATH,
                                       seed=seed)  # <--- This makes the files permanent
                so.optimize()
