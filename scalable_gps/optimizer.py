import torch
from torch import Tensor
from objective import OptimizationProblem
from typing import Optional, Any, List, Dict
from turbo_state import TurboInstance
from objective import Ackley
from scalable_gps.objective import Ackley
from scalable_gps.turbo_state import TurboInstance
from pyspark.sql import SparkSession


class ScalableOptimizer:

    def __init__(self,
                 objective: OptimizationProblem,
                 num_parallel: int = 2,
                 num_total_iterations: int = -1,
                 batch_size: int = 3,
                 turbo_kwargs: Optional[Dict] = {},
                 model_kwargs: Optional[Dict] = {}
    ):

        #self.spark = SparkSession.builder.getOrCreate()
        #self.sc = self.spark.sparkContext
        self.batch_size = batch_size
        self.num_total_iterations = num_total_iterations
        self.turbo_kwargs = turbo_kwargs
        self.model_kwargs = model_kwargs

        self.turbo_processes = [TurboInstance(
            batch_size=batch_size, function=objective, identifier=f"TR-{i}", **turbo_kwargs)  for i in range(num_parallel)]
        self.objective = objective

    def optimize(self):

        #turbos = self.sc.parallelize(self.turbo_processes)
        #turbos.foreach(lambda t: t.optimize())
        all_X = torch.empty((0, objective.dim()))
        all_y = torch.empty(0)
        covar_module = None
        
        for proc in self.turbo_processes:
            res_X, res_y = proc.optimize()
            all_X = torch.cat((all_X, res_X))
            all_y = torch.cat((all_y, res_y))
            
        print(all_X, all_y)


if __name__ == '__main__':
    objective = Ackley(4)
    so = ScalableOptimizer(objective, num_parallel=4)
    so.optimize()
