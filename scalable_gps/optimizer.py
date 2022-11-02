import torch
from torch import Tensor
from objective import OptimizationProblem
from typing import Optional, Any, List, Dict
from turbo_state import TurboInstance
from objective import Ackley
from scalable_gps.objective import Ackley
from scalable_gps.turbo_state import TurboInstance
from pyspark.sql import SparkSession

import tqdm
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel, GridInterpolationKernel

from gpytorch.mlls import MarginalLogLikelihood, ExactMarginalLogLikelihood
from dkl_model import FeatureExtractor, GPRegressionModel

class ScalableOptimizer:

    def __init__(self,
                 objective: OptimizationProblem,
                 num_parallel: int = 2,
                 num_total_iterations: int = -1,
                 batch_size: int = 2,
                 turbo_kwargs: Optional[Dict] = {},
    ):

        self.spark = SparkSession.builder.getOrCreate()
        self.sc = self.spark.sparkContext
        self.batch_size = batch_size
        self.num_total_iterations = num_total_iterations
        self.turbo_kwargs = turbo_kwargs
        self.num_parallel = num_parallel
        self.batch_size = batch_size
        self.turbo_kwargs = turbo_kwargs
        self.objective = objective

    def optimize(self):
        self.turbo_processes = [TurboInstance(
        batch_size=self.batch_size, function=objective, identifier=f"TR-{i}", **turbo_kwargs)  for i in range(self.num_parallel)]
        turbos = self.sc.parallelize(self.turbo_processes)
        res = turbos.map(lambda t: t.optimize())
        data = res.collect()
        all_X = torch.cat([Tensor(proc_data[0]) for proc_data in data], axis=0)
        all_y = torch.cat([Tensor(proc_data[1]) for proc_data in data], axis=0)
        train_y = (all_y - all_y.mean()) / all_y.std()
            
        dk_model = self._train_deepkernel(all_X, train_y)
        posterior = dk_model.posterior(all_X)  
        print('prediction errors', posterior.mean.flatten() - train_y.flatten())
        from botorch.generation import MaxPosteriorSampling
        thompson_sampling = MaxPosteriorSampling(model=dk_model, replacement=False)

        from torch.quasirandom import SobolEngine
        sobol = SobolEngine(4, scramble=True)
        X_cand = sobol.draw(1000)
        
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=10)
            print('Testing DKL TS')
            print(X_next)
            print(objective(X_next))

    def _train_deepkernel(self, X: Tensor, y: Tensor, num_iters: int = 100):
        likelihood = GaussianLikelihood()
        feature_extractor = FeatureExtractor(objective.dim())
        dkl_model = GPRegressionModel(X, y, likelihood, feature_extractor)
        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()
        
        optimizer = torch.optim.Adam([
            {'params': dkl_model.feature_extractor.parameters()},
            {'params': dkl_model.covar_module.parameters()},
            {'params': dkl_model.mean_module.parameters()},
            {'params': dkl_model.likelihood.parameters()},
        ], lr=0.01)
        mll = ExactMarginalLogLikelihood(likelihood, dkl_model)
        iterator = tqdm.tqdm(range(num_iters))
        # on-the-fly SGD - should probably be implemented according to a paper on overfit in DKL
        for i in iterator:
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = dkl_model(X)
            # Calc loss and backprop derivatives
            loss = -mll(output, y)
            
            loss.backward()
            iterator.set_postfix(loss=loss.item())
            optimizer.step()
        return dkl_model

if __name__ == '__main__':


    objective = Ackley(4)
    turbo_kwargs = {
        'model': SingleTaskGP,      
    }
    so = ScalableOptimizer(objective, num_parallel=4, turbo_kwargs=turbo_kwargs)
    so.optimize()
