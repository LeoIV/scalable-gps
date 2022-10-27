import numpy as np
from pyspark.ml.linalg import DenseVector, DenseMatrix
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow
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


def gram_matrix(x: RDD[DenseVector], y: RDD[DenseVector], lengthscales: DenseVector) -> IndexedRowMatrix:
    """
    Compute the gram matrix for a given set of vector RDDs and a given kernel function

    Args:
        x: RDD of DenseVectors, vectors all need to have the same dimensionality (as y)
        y: RDD of DenseVectors, vectors all need to have the same dimensionality (as x)
        lengthscales: the lengthscales, have to have the same dimensionality as x and y or 1D

    Returns: Gram matrix for the given kernel function

    """
    sc = x.context

    X = np.array(x.map(lambda x: x.toArray()).collect())
    Y = np.array(y.map(lambda y: y.toArray()).collect())

    kern = RBF(lengthscales.toArray())
    sigma = kern(X, Y)

    cov_mtrx = IndexedRowMatrix(
        numRows=sigma.shape[0],
        numCols=sigma.shape[1],
        rows=sc.parallelize(sigma).zipWithIndex().map(lambda r: IndexedRow(r[1], r[0]))
    )

    return cov_mtrx


def matrix_inverse(matrix: IndexedRowMatrix) -> DenseMatrix:
    """
    Compute the inverse of the given coordinate matrix using singular value decomposition.

    Args:
        matrix: the matrix to invert

    Returns: the inverted matrix

    """

    svd = matrix.computeSVD(k=matrix.numCols(), computeU=True, rCond=1e-15)  # do the SVD

    s_diag_inverse = 1 / svd.s.toArray()
    u = np.array(svd.U.rows.sortBy(lambda r: r.index).map(lambda r: r.vector.toArray()).collect())
    _matrix_inverse = np.matmul(svd.V.toArray(), np.multiply(s_diag_inverse[:, np.newaxis], u.T))
    return DenseMatrix(
        numRows=_matrix_inverse.shape[0],
        numCols=_matrix_inverse.shape[1],
        values=_matrix_inverse.ravel(order="F")  # Spark is column major, numpy by default row major
    )  # return inverse as dense matrix
