import numpy as np
from pyspark.ml.linalg import DenseVector, DenseMatrix
from pyspark.mllib.linalg.distributed import CoordinateMatrix, Vector
from pyspark.rdd import RDD
from pyspark.context import SparkContext

from scalable_gps.linalg import gram_matrix, matrix_inverse


class GaussianProcess:

    def __init__(self, x: RDD[DenseVector], y: RDD[float], lengthscales: DenseVector, sc: SparkContext):
        self.x = x
        self.y = y
        self.x.cache()
        self.y.cache()
        self.y_arr = np.array(self.y.collect())
        self.lengthscales = lengthscales
        self.gram = gram_matrix(self.x, self.x, self.lengthscales, sc)
        self.inv = matrix_inverse(self.gram)
        self.sc = sc

    @staticmethod
    def coordinate_to_dense_matrix(matrix: CoordinateMatrix) -> DenseMatrix:
        return DenseMatrix(
            numRows=matrix.numRows(),
            numCols=matrix.numCols(),
            values=np.array(matrix.toRowMatrix().rows.map(lambda x: x.toArray()).collect()).ravel(order="F")
        )

    def predict_mean_var(self, x: RDD[DenseVector]):
        gram_1 = gram_matrix(x, self.x, self.lengthscales, self.sc)
        gram_2 = np.array(
            gram_matrix(  # speedup: we only need the diagonal here, make gram matrix support that
                x, x,
                self.lengthscales,
                self.sc)
            .rows
            .zipWithIndex()
            .map(lambda x: x[0][x[1]])  # extract only the diagonal elements
            .collect())
        gram_1_dense = np.array(gram_1.rows.map(lambda x: x.toArray()).collect())
        mean = gram_1_dense @ self.inv.toArray() @ self.y_arr
        var = gram_2 - np.diagonal(gram_1_dense @ self.inv.toArray() @ gram_1_dense.T)
        return mean, var
