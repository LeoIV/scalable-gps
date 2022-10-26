import numpy as np
from pyspark.ml.linalg import DenseVector, DenseMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix
from pyspark.rdd import RDD


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
    return np.exp(np.sum((x.toArray() - y.toArray()) ** 2 / (2 * lengthscales.toArray()) ** 2))


def gram_matrix(x: RDD[DenseVector], y: RDD[DenseVector], lengthscales: DenseVector,
                kernel: str = "rbf-ard") -> CoordinateMatrix:
    """
    Compute the gram matrix for a given set of vector RDDs and a given kernel function

    Args:
        x: RDD of DenseVectors, vectors all need to have the same dimensionality (as y)
        y: RDD of DenseVectors, vectors all need to have the same dimensionality (as x)
        lengthscales: the lengthscales, have to have the same dimensionality as x and y

    Returns: Gram matrix for the given kernel function

    """

    match kernel:
        case "rbf-ard":
            kern_func = rbf_ard_kernel
        case _:
            raise RuntimeError("Unsupported kernel function")

    x_indexed = x.zipWithIndex()
    y_indexed = y.zipWithIndex()
    cartesian = x_indexed.cartesian(y_indexed)
    cov_indexed = cartesian.map(lambda x: MatrixEntry(x[0][1], x[1][1], kern_func(x[0][0], x[1][0], lengthscales)))

    cov_mtrx = CoordinateMatrix(cov_indexed)

    return cov_mtrx


def matrix_inverse(matrix: CoordinateMatrix) -> DenseMatrix:
    """
    Compute the inverse of the given coordinate matrix using singular value decomposition.

    Args:
        matrix: the matrix to invert

    Returns: the inversed matrix

    """
    k = min(matrix.numRows(), matrix.numCols())  # number of singular values
    svd = matrix.toRowMatrix().computeSVD(k=k, computeU=True)  # do the SVD
    u_dense = DenseMatrix(numRows=svd.U.numRows(), numCols=svd.U.numCols(),
                          values=svd.U.rows.flatMap(
                              lambda x: x).collect())  # get U as a dense matrix (this performs actions!)
    s_inv = np.diag(1 / svd.s)  # invert S and make diagonal matrix
    cov_inv = svd.V.toArray() @ s_inv @ u_dense.toArray().T  # compute inverse
    return DenseMatrix(numRows=cov_inv.shape[0], numCols=cov_inv.shape[1],
                       values=cov_inv.ravel())  # return inverse as dense matrix
