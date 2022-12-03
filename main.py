from botorch.models import SingleTaskGP

from scalable_gps.objective import Ackley
from scalable_gps.optimizer import ScalableOptimizer

if __name__ == '__main__':
    objective = Ackley(4)
    turbo_kwargs = {
        'model': SingleTaskGP,
    }
    so = ScalableOptimizer(objective, num_parallel=4, turbo_kwargs={})
    so.optimize()
