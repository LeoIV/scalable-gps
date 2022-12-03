from logging import info
from typing import Optional, Dict

import torch
import tqdm
import gpytorch
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.pipeline import Pipeline

from torch import Tensor
from torch.quasirandom import SobolEngine

from sparktorch import serialize_torch_obj, SparkTorch
from pyspark.ml.feature import VectorAssembler

from dkl_model import FeatureExtractor, DeepKernelGPRegressor
from objective import OptimizationProblem
from scalable_gps.objective import Ackley
from scalable_gps.turbo_state import TurboInstance
from utils import save


class ScalableOptimizer:

    def __init__(self,
                 objective: OptimizationProblem,
                 outer_iterations: int = 10,
                 num_parallel: int = 2,
                 num_total_iterations: int = -1,
                 batch_size: int = 2,
                 turbo_kwargs: Optional[Dict] = {},
                 name: str = 'TurBO-DKL'
                 ):
        self.spark = SparkSession.builder.getOrCreate()
        self.sc = self.spark.sparkContext
        self.batch_size = batch_size
        self.num_total_iterations = num_total_iterations
        self.num_parallel = num_parallel
        self.batch_size = batch_size
        self.turbo_kwargs = turbo_kwargs
        self.objective = objective
        self.dim = objective.dim()
        self.outer_iterations = outer_iterations
        self.name = name

    def optimize(self):
        x_global = torch.empty((0, self.dim))
        y_global = torch.empty(0)

        deep_kernel_model = None
        for i_outer in range(self.outer_iterations):
            info(f"Starting outer iteration {i_outer + 1}")
            self.turbo_processes = [
                TurboInstance(
                    batch_size=self.batch_size,
                    function=objective,
                    model=deep_kernel_model if deep_kernel_model is not None else SingleTaskGP,
                    identifier=f"TR-{i}")
                for i in range(self.num_parallel)
            ]
            turbos = self.sc.parallelize(self.turbo_processes)
            res = turbos.map(lambda t: t.optimize())
            data = res.collect()
            x_aggregated = torch.cat([Tensor(proc_data[0])
                                     for proc_data in data], axis=0)
            y_aggregated = torch.cat([Tensor(proc_data[1])
                                     for proc_data in data], axis=0)
            x_global = torch.cat((x_global, x_aggregated), dim=0)
            y_global = torch.cat((y_global, y_aggregated), dim=0)
            y_global_normalized = (y_global - y_global.mean()) / y_global.std()

            deep_kernel_model = self._train_deepkernel(
                x_global, y_global_normalized)

        deep_kernel_model_posterior = deep_kernel_model.posterior(x_global)
        print('prediction errors',
              deep_kernel_model_posterior.mean.flatten() - ((y_global - y_global.mean()) / y_global.std()).flatten())
        thompson_sampling = MaxPosteriorSampling(
            model=deep_kernel_model, replacement=False)
        sobol = SobolEngine(self.dim, scramble=True)
        X_cand = sobol.draw(1000)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=10)

        save(x_global, y_global, self.name, objective.name())

    def _train_deepkernel(self, X: Tensor, y: Tensor, num_iters: int = 100):
        likelihood = GaussianLikelihood()
        feature_extractor = FeatureExtractor(objective.dim())
        dkl_model = DeepKernelGPRegressor(X, y, likelihood, feature_extractor)

        optimizer = torch.optim.Adam

        # make all the parameter generators into lists
        parameters = [{'params': [p for p in dkl_model.feature_extractor.parameters()]},
                      {'params': [
                          p for p in dkl_model.covar_module.parameters()]},
                      {'params': [p for p in dkl_model.mean_module.parameters()]},
                      {'params': [p for p in dkl_model.likelihood.parameters()]},
                      ]
        mll = ExactMarginalLogLikelihood(likelihood, dkl_model)
        torch_obj = serialize_torch_obj(
            model=dkl_model,
            # need to return a scalar, returns a vector of inidividual losses
            criterion=lambda output, y_train: mll(output, y_train).sum(),
            optimizer=optimizer,
            lr=0.01
        )
        data = self.sc.parallelize(
            torch.cat((y.unsqueeze(-1), X), axis=1).detach().numpy().tolist())
        df = data.toDF()
        vector_assembler = VectorAssembler(
            inputCols=df.columns[1:self.dim+1], outputCol='features')
        # Setup features

        stm = SparkTorch(
            inputCol='features',
            labelCol='_1',  # this tells SparkTorch to consider the first column as the label column
            predictionCol='predictions',
            torchObj=torch_obj,
            verbose=0,
            iters=3,
            miniBatch=16
        )

        # Can be used in a pipeline and saved.
        p = Pipeline(stages=[vector_assembler, stm]).fit(df)
        pt_model = p.stages[1].getPytorchModel()
        return pt_model


if __name__ == '__main__':
    objective = Ackley(4)
    turbo_kwargs = {
        'model': SingleTaskGP,
    }
    so = ScalableOptimizer(objective, num_parallel=4, turbo_kwargs={})
    so.optimize()
