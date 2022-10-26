import numpy as np
from pyspark.context import SparkContext
from pyspark.ml.linalg import DenseVector, DenseMatrix
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.rdd import RDD
from sklearn.gaussian_process.kernels import RBF


def rbf_ard_kernel(x: DenseVector, y: DenseVector, lengthscales: DenseVector):
    """
    RBF kernel function with automated relevance detection

    Args:
        x: vector 1
        y: vector 2
        lengthscales: lengthscales, same dimensionality as x and y

    Returns: k(x,y)

    """
    assert len(x) == len(y)
    assert x.ndim == y.ndim
    assert x.ndim == 1
    assert len(lengthscales) == len(y)
    assert lengthscales.ndim == y.ndim
    assert lengthscales.ndim == 1
    return np.exp(np.sum((x.toArray() - y.toArray()) ** 2 / ((2 * lengthscales.toArray()) ** 2)))


def gram_matrix(x: RDD[DenseVector], y: RDD[DenseVector], lengthscales: DenseVector, sc: SparkContext) -> RowMatrix:
    """
    Compute the gram matrix for a given set of vector RDDs and a given kernel function

    Args:
        x: RDD of DenseVectors, vectors all need to have the same dimensionality (as y)
        y: RDD of DenseVectors, vectors all need to have the same dimensionality (as x)
        lengthscales: the lengthscales, have to have the same dimensionality as x and y or 1D

    Returns: Gram matrix for the given kernel function

    """

    X = np.array(x.map(lambda x: x.toArray()).collect())
    Y = np.array(y.map(lambda y: y.toArray()).collect())

    kern = RBF(lengthscales.toArray())
    sigma = kern(X, Y)

    cov_mtrx = RowMatrix(numRows=sigma.shape[0], numCols=sigma.shape[1],
                         rows=sc.parallelize(sigma))

    return cov_mtrx


def matrix_inverse(matrix: RowMatrix) -> DenseMatrix:
    """
    Compute the inverse of the given coordinate matrix using singular value decomposition.

    Args:
        matrix: the matrix to invert

    Returns: the inverted matrix

    """

    svd = matrix.computeSVD(k=matrix.numCols(), computeU=True, rCond=1e-15)  # do the SVD

    s_inv = 1 / svd.s
    mtrx_orig = np.array(matrix.rows.map(lambda x: x.toArray()).collect())
    u_dense = mtrx_orig @ (svd.V.toArray() * s_inv[np.newaxis, :])
    cov_inv = np.matmul(svd.V.toArray(), np.multiply(s_inv[:, np.newaxis], u_dense.T))
    return DenseMatrix(numRows=cov_inv.shape[0], numCols=cov_inv.shape[1],
                       values=cov_inv.ravel(order="F"))  # return inverse as dense matrix
