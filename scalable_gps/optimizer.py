import torch
from torch import Tensor
from objective import OptimizationProblem
from typing import Optional, Any, List, Dict
from turbo_state import TurboInstance
from objective import Ackley
from scalable_gps.objective import Ackley
from scalable_gps.turbo_state import TurboInstance
from pyspark.sql import SparkSession

from botorch.models import SingleTaskGP
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.mlls import MarginalLogLikelihood, ExactMarginalLogLikelihood


class ScalableOptimizer:

    def __init__(self,
                 objective: OptimizationProblem,
                 num_parallel: int = 2,
                 num_total_iterations: int = -1,
                 batch_size: int = 4,
                 turbo_kwargs: Optional[Dict] = {},
    ):

        self.spark = SparkSession.builder.getOrCreate()
        self.sc = self.spark.sparkContext
        self.batch_size = batch_size
        self.num_total_iterations = num_total_iterations
        self.turbo_kwargs = turbo_kwargs
        
        self.turbo_processes = [TurboInstance(
            batch_size=batch_size, function=objective, identifier=f"TR-{i}", **turbo_kwargs)  for i in range(num_parallel)]
        self.objective = objective

    def optimize(self):

        turbos = self.sc.parallelize(self.turbo_processes)
        res = turbos.map(lambda t: t.optimize())
        
        data = res.collect()

if __name__ == '__main__':
    objective = Ackley(4)
    turbo_kwargs = {
        'model': SingleTaskGP,
        
    }
    so = ScalableOptimizer(objective, num_parallel=4, turbo_kwargs=turbo_kwargs)
    so.optimize()
