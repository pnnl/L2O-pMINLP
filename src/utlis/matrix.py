"""
Matrx processing
"""

import numpy as np

def identityTransform(A, m):
    """
    Construct a modified null space basis to forming an identity block
    corresponding to integer variables, using column transformations.
    """
    A_transformed = A.copy()
    pivot_col = 0
    # Column transformations to form the identity block in the first m rows
    for pivot_row in range(m):
        # find the pivot column with the largest absolute value in the current pivot position
        max_col = pivot_col + np.argmax(np.abs(A_transformed[pivot_row, pivot_col:]))
        # swap columns to bring the pivot column to the current pivot position
        A_transformed[:, [pivot_col, max_col]] = A_transformed[:, [max_col, pivot_col]]
        # normalize the pivot column so that the pivot element becomes 1
        A_transformed[:, pivot_col] = A_transformed[:, pivot_col] / A_transformed[pivot_row, pivot_col]
        # eliminate entries in the pivot row for all other rows
        mask = np.arange(A.shape[1]) != pivot_col
        factors = A_transformed[pivot_row, mask]
        A_transformed[:, mask] -= np.outer(A_transformed[:, pivot_col], factors)
        # move to the next pivot column
        pivot_col += 1
    return A_transformed
