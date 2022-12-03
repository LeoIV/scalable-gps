import math
from dataclasses import dataclass
from typing import Optional, Dict, List

import gpytorch
import torch
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.mlls import MarginalLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch import Tensor
from torch.quasirandom import SobolEngine

from scalable_gps.objective import OptimizationProblem
from scalable_gps.turbo_util import generate_batch


class TurboInstance:

    def __init__(
            self,
            batch_size: int,
            function: OptimizationProblem,
            model: ExactGP,
            model_kwargs: Dict = {},
            likelihood: Likelihood = GaussianLikelihood,
            model_parameters: List[Dict[str, Tensor]] = None,
            mll_opt: MarginalLogLikelihood = ExactMarginalLogLikelihood,
            n_init: Optional[int] = None,
            identifier: str = "",
            seed: int = 0
    ):
        self.batch_size = batch_size
        self.dim = function.dim()
        self.function = function
        self.model = model
        self.model_kwargs = model_kwargs
        self.likelihood = likelihood
        self.parameters = model_parameters
        self.mll_opt = mll_opt
        self.n_init = 2 * self.dim if n_init is None else n_init
        self.state = TurboState(self.dim, self.batch_size)
        self.num_restarts = 10
        self.raw_samples = 512
        self.n_candidates = min(5000, max(2000, 200 * self.dim))
        self.identifier = identifier
        self.seed = seed

        self.X = torch.empty((0, self.dim))
        self.y = torch.empty(0)
        self.has_run = False

    def get_initial_points(self):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        X_init = sobol.draw(n=self.n_init)
        return X_init

    def optimize(self):
        x_init = self.get_initial_points()
        y_init = torch.tensor([self.function(x) for x in x_init])

        self.X = torch.cat((self.X, x_init), dim=0)
        self.y = torch.cat((self.y, y_init))

        while not self.state.restart_triggered:
            train_y = (self.y - self.y.mean()) / self.y.std()

            model = self.model(
                self.X, train_y.unsqueeze(-1), **self.model_kwargs) if not hasattr(self.model,
                                                                                   'feature_extractor') else self.model

            with gpytorch.settings.max_cholesky_size(float("inf")):
                # Fit the model
                if not hasattr(self.model, 'feature_extractor'):
                    mll = self.mll_opt(model.likelihood, model)
                    fit_gpytorch_torch(mll, options={'disp': False})

                # Create a batch
                x_next = generate_batch(
                    state=self.state,
                    model=model,
                    X=self.X,
                    Y=train_y,
                    batch_size=self.batch_size,
                    n_candidates=self.n_candidates,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    acqf="ts",
                )

                y_next = torch.tensor([self.function(x) for x in x_next])

                self.state = update_state(self.state, y_next)

                self.X = torch.cat((self.X, x_next), dim=0)
                self.y = torch.cat((self.y, y_next), dim=0)
                print(
                    f"{self.identifier}: {len(self.X)}) Best value: {self.state.best_value:.3}, TR length: {self.state.length:.3f}"
                )
        self.has_run = True
        return self.X, self.y

    def reset(self):
        pass


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 2
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, y_next):
    if max(y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state
