import logging
from typing import Union

import numpy as np
from pyspark.ml.linalg import DenseVector, DenseMatrix
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow
from pyspark.rdd import RDD

from scalable_gps.util import _to_list, _append, _extend, _sort_row


def rbf_ard_kernel(x: DenseVector, y: DenseVector, lengthscales: DenseVector):
    """
    RBF kernel function with automated relevance detection

    Args:
        x: vector 1
        y: vector 2
        lengthscales: lengthscales, same dimensionality as x and y

    Returns: k(x,y)

    """
    _x = (x / lengthscales).toArray()
    _y = (y / lengthscales).toArray()

    dist = np.sum((_x - _y) ** 2)

    return np.exp(-0.5 * dist)


def gram_matrix(x: RDD[DenseVector], y: RDD[DenseVector], lengthscales: DenseVector, diagonal=False) -> \
        Union[IndexedRowMatrix, DenseVector]:
    """
    Compute the gram matrix for a given set of vector RDDs and a given kernel function.
    y is ignored if diagonal is True.

    Args:
        x: RDD of DenseVectors, vectors all need to have the same dimensionality (as y)
        y: RDD of DenseVectors, vectors all need to have the same dimensionality (as x)
        lengthscales: the lengthscales, have to have the same dimensionality as x and y or 1D
        diagonal: only compute diagonal elements, y is ignored if diagonal is True

    Returns: Gram matrix for the given kernel function or DenseVector if diagonal is True

    """
    if diagonal:
        logging.info("for diagonal, ignoring y")
        matrix_entries = x.map(lambda v: rbf_ard_kernel(v, v, lengthscales))
        return DenseVector(matrix_entries.collect())

    x_indexed_rows = x.zipWithIndex().map(lambda r: IndexedRow(r[1], r[0]))
    y_indexed_rows = y.zipWithIndex().map(lambda r: IndexedRow(r[1], r[0]))

    xy = x_indexed_rows.cartesian(y_indexed_rows)
    matrix_entries = xy.map(
        lambda rows: (
            rows[0].index,
            (rows[1].index,
             rbf_ard_kernel(rows[0].vector, rows[1].vector, lengthscales))
        )
    )

    matrix_rows = matrix_entries.combineByKey(_to_list, _append, _extend)
    matrix_rows = matrix_rows.map(lambda ir: IndexedRow(index=ir[0], vector=_sort_row(ir[1])))
    matrix_rows.cache()

    matrix = IndexedRowMatrix(matrix_rows)

    return matrix


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
