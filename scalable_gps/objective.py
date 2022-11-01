from abc import ABC, abstractmethod
import torch
import numpy as np
from botorch.test_functions import Ackley as _Ackley
from botorch.utils.transforms import unnormalize


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


class Ackley(OptimizationProblem):

    def __init__(self, dim: int):
        self._dim = dim
        self._ackley = _Ackley(dim=dim, negate=True)

    def __call__(self, x: torch.Tensor):
        return eval_objective(x, self._ackley)

    def lb(self) -> np.ndarray:
        return self._ackley.bounds[0]

    def ub(self) -> np.ndarray:
        return self._ackley.bounds[1]

    def dim(self) -> int:
        return self._dim
