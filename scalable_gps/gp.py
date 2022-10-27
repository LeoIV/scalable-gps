import numpy as np
from pyspark.ml.linalg import DenseVector
from pyspark.rdd import RDD

from scalable_gps.linalg import gram_matrix, matrix_inverse


class GaussianProcess:

    def __init__(self, x: RDD[DenseVector], y: RDD[float], lengthscales: DenseVector):
        self.x = x
        self.y = y
        self.x.cache()
        self.y.cache()
        self.y_arr = np.array(self.y.collect())
        self.lengthscales = lengthscales
        self.gram = gram_matrix(self.x, self.x, self.lengthscales)
        self.inv = matrix_inverse(self.gram)

    def predict_mean_var(self, x: RDD[DenseVector]):
        gram_1 = gram_matrix(x, self.x, self.lengthscales)
        gram_2 = np.array(
            gram_matrix(  # speedup: we only need the diagonal here, make gram matrix support that
                x, x,
                self.lengthscales)
            .rows
            .sortBy(lambda r: r.index)
            .map(lambda x: x.vector[x.index])  # extract only the diagonal elements
            .collect())
        gram_1_dense = np.array(
            gram_1
            .rows
            .sortBy(lambda r: r.index)
            .map(lambda x: x.vector.toArray())
            .collect())
        mean = gram_1_dense @ self.inv.toArray() @ self.y_arr
        var = gram_2 - np.diagonal(gram_1_dense @ self.inv.toArray() @ gram_1_dense.T)
        return mean, var
