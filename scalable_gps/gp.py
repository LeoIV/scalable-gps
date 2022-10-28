from typing import Tuple

import numpy as np
from pyspark.ml.linalg import DenseVector, DenseMatrix
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow
from pyspark.rdd import RDD

from scalable_gps.linalg import matrix_inverse, gram_matrix


class GaussianProcess:

    def __init__(
            self,
            x: RDD[DenseVector],
            y: RDD[float],
            lengthscales: np.ndarray,
            noise: float,
            signal_variance: float
    ):
        self.x = x
        self.y = y
        self.x.cache()
        self.y.cache()
        self.y_arr = np.array(self.y.collect())

        # private members
        self._lengthscales = lengthscales
        self._noise = noise
        self._signal_variance = signal_variance
        self._gram = None
        self._gram_noisy = None
        self._inv = None
        self._gram_dense = None
        self._y_dense = None

    def _reset(self) -> None:
        """
        Reset pre-computed matrices since something changed

        Returns: None

        """
        self._gram_noisy = None
        self._inv = None

    @property
    def noise(self) -> float:
        return self._noise

    @noise.setter
    def noise(self, noise) -> None:
        self._reset()
        self._noise = noise

    @property
    def signal_variance(self) -> float:
        return self._signal_variance

    @signal_variance.setter
    def signal_variance(self, signal_variance: float) -> None:
        self._reset()
        self._signal_variance = signal_variance

    @property
    def lengthscales(self) -> np.ndarray:
        return self._lengthscales

    @lengthscales.setter
    def lengthscales(self, lengthscales: np.ndarray) -> None:
        self._reset()
        self._lengthscales = lengthscales

    @property
    def gram(self) -> IndexedRowMatrix:
        if self._gram is None:
            self._gram = gram_matrix(self.x, self.x, DenseVector(self.lengthscales), self.signal_variance)
        return self._gram

    @property
    def gram_with_noise(self) -> IndexedRowMatrix:
        if self._gram_noisy is None:
            self._gram_noisy = self._gram_with_noise(self.noise)
        return self._gram_noisy

    @property
    def inv(self) -> DenseMatrix:
        if self._inv is None:
            self._inv = matrix_inverse(self.gram_with_noise)
        return self._inv

    @property
    def gram_dense(self) -> np.ndarray:
        if self._gram_dense is None:
            self._gram_dense = np.array(
                self.gram.rows.sortBy(lambda r: r.index).map(lambda r: r.vector.toArray()).collect())
        else:
            return self._gram_dense

    @property
    def y_dense(self) -> np.ndarray:
        return np.array(self.y.collect())

    def _gram_with_noise(self, noise: float):
        """
        The gram matrix with noise

        Args:
            noise: noise to add

        Returns: the gram matrix with noise

        """
        irm = IndexedRowMatrix(
            self.gram.rows.map(
                lambda r: IndexedRow(
                    r.index,
                    r.vector + noise * (np.arange(len(r.vector)) == r.index).astype(np.float64))
            )
        )
        irm.rows.cache()
        return irm

    def predict_mean_var(self, x: RDD[DenseVector]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return mean and variance for a set of points

        Args:
            x: points to predict mean and variance for

        Returns: tuple[np.ndarray, np.ndarray] of mean and variance, same length as x

        """
        gram_1 = gram_matrix(x, self.x, DenseVector(self.lengthscales), self.signal_variance)
        gram_2 = gram_matrix(  # speedup: we only need the diagonal here, make gram matrix support that
            x, x,
            DenseVector(self.lengthscales),
            self.signal_variance,
            diagonal=True
        ).toArray()
        gram_1_dense = np.array(
            gram_1
            .rows
            .sortBy(lambda r: r.index)
            .map(lambda x: x.vector.toArray())
            .collect())
        mean = gram_1_dense @ self.inv.toArray() @ self.y_arr
        var = gram_2 - np.diagonal(gram_1_dense @ self.inv.toArray() @ gram_1_dense.T)
        return mean, var

    def negative_log_likelihood(self):
        """

        Returns: negative log likelihood for the model

        """

        if self.gram_with_noise.numRows() <= 2 ** 10 and self.gram_with_noise.numCols() <= 2 ** 10:
            gwn_dense = self.gram_with_noise.toBlockMatrix().blocks.first()[1].toArray()
        else:
            gwn_dense = np.array(self.gram_with_noise.rows.sortBy(lambda r: r.index).map(
                lambda r: r.vector.toArray()).collect())

        sign, ldet = np.linalg.slogdet(gwn_dense)
        return 0.5 * (sign * ldet + self.y_dense.dot(
            self.inv.toArray().dot(self.y_dense)) + self.x.count() * np.log(
            2 * np.pi))

    def optimize(
            self,
            lengthscale_constraints: Tuple[np.ndarray, np.ndarray],
            noise_constraints: Tuple[float, float],
            signal_variance_constraints: Tuple[float, float],
            method: str = "random search",
            n_steps: int = 10,
            grid_resolution: int = 100
    ):

        ls_range = np.linspace(lengthscale_constraints[0], lengthscale_constraints[1], grid_resolution).T
        nc_range = np.linspace(noise_constraints[0], noise_constraints[1], grid_resolution)
        sv_range = np.linspace(signal_variance_constraints[0], signal_variance_constraints[1], grid_resolution)

        rng = np.random.default_rng()
        param_configs = rng.choice(np.vstack([ls_range, nc_range, sv_range]), size=n_steps, axis=1).T

        nlls = []

        for i, param_config in enumerate(param_configs):
            ls = param_config[:-2]
            noise = param_config[-2]
            signal_variance = param_config[-1]

            self.lengthscales = ls
            self.noise = noise
            self.signal_variance = signal_variance

            nll = self.negative_log_likelihood()
            nlls.append(nll)

        best_param = np.argmin(nlls)

        ls = param_configs[best_param, :-2]
        noise = param_configs[best_param, -2]
        signal_variance = param_configs[best_param, -1]

        print(f"Minimum negative log likelihood of {np.min(nlls):.3}, noise is set to {noise:.3} and signal variance "
              f"to {signal_variance:.3}")

        self.lengthscales = ls
        self.noise = noise
        self.signal_variance = signal_variance
