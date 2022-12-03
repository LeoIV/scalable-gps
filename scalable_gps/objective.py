from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np
from botorch.test_functions import Ackley as _Ackley, Griewank as _Griewank
from botorch.utils.transforms import unnormalize

from scalable_gps.benchmark_utils import RoverDomain, create_large_domain


def eval_objective(x, fun):
    """This is a helper function we use to unnormalize and evaluate a point"""
    return fun(unnormalize(x, fun.bounds))


class OptimizationProblem(ABC):

    @abstractmethod
    def __init__(self, dim: int):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, x: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def lb(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def ub(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()


class Ackley(OptimizationProblem):

    def __init__(self, dim: int):
        self._dim = dim
        self._ackley = _Ackley(dim=dim, negate=True)
        self._name = f'Ackley-{dim}'

    def __call__(self, x: torch.Tensor):
        return eval_objective(x, self._ackley)

    def lb(self) -> np.ndarray:
        return self._ackley.bounds[0]

    def ub(self) -> np.ndarray:
        return self._ackley.bounds[1]

    def dim(self) -> int:
        return self._dim

    def name(self) -> str:
        return self._name


class Griewank(OptimizationProblem):

    def __init__(self, dim: int):
        self._dim = dim
        self._griewank = _Griewank(dim=dim, negate=True)

    def __call__(self, x: torch.Tensor):
        return eval_objective(x, self._griewank)

    def lb(self) -> np.ndarray:
        return self._griewank.bounds[0]

    def ub(self) -> np.ndarray:
        return self._griewank.bounds[1]

    def dim(self) -> int:
        return self._dim


class Rover(OptimizationProblem):

    def __init__(self, dim: int):
        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)

        domain: RoverDomain = create_large_domain(
            force_start=False,
            force_goal=False,
            start_miss_cost=l2cost,
            goal_miss_cost=l2cost,
        )
        self._domain = domain

    def __call__(self, x: torch.Tensor):
        x = x.numpy()
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        costs = [-self._domain(y) for y in x]

        costs = np.array(costs).squeeze()
        return torch.tensor(costs)

    def lb(self) -> np.ndarray:
        return torch.zeros(60)

    def ub(self) -> np.ndarray:
        return torch.ones(60)

    def dim(self) -> int:
        return 60
