from botorch.models import SingleTaskGP


from scalable_gps.objective import Ackley, Griewank, Rover
from scalable_gps.optimizer import ScalableOptimizer

if __name__ == '__main__':
    objectives = [Rover(60)]
    for objective in objectives:
        for use_dkl in [True, False]:
            for rep in range(1):
                name = "TurBO-DKL" if use_dkl else "TurBO"
                turbo_kwargs = {
                    'model': SingleTaskGP,
                }
                so = ScalableOptimizer(objective, num_parallel=4, turbo_kwargs={}, name=name, use_dkl=use_dkl, outer_iterations=2)
                so.optimize()