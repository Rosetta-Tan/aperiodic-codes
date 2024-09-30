"""
Assorted functions to work with binary vectors and matrices
Copyright: Joschka Roffe, 2021
"""

import numpy as np
from numba import njit

@njit('(int64[:,::1], bool_)', cache=True)
def row_echelon(matrix, full):
    
    """
    Converts a binary matrix to row echelon form via Gaussian Elimination

    Parameters
    ----------
    matrix : numpy.ndarry
        A binary matrix in numpy.ndarray format
    full: bool, optional
        If set to `True', Gaussian elimination is only performed on the rows below
        the pivot. If set to `False' Gaussian eliminatin is performed on rows above
        and below the pivot. 
    
    Returns
    -------
        row_ech_form: numpy.ndarray
            The row echelon form of input matrix
        rank: int
            The rank of the matrix
        transform_matrix: numpy.ndarray
            The transformation matrix such that (transform_matrix@matrix)=row_ech_form
        pivot_cols: list
            List of the indices of pivot num_cols found during Gaussian elimination

    Examples
    --------
    >>> H=np.array([[1, 1, 1],[1, 1, 1],[0, 1, 0]])
    >>> re_matrix=row_echelon(H)[0]
    >>> print(re_matrix)
    [[1 1 1]
     [0 1 0]
     [0 0 0]]

    >>> re_matrix=row_echelon(H,full=True)[0]
    >>> print(re_matrix)
    [[1 0 1]
     [0 1 0]
     [0 0 0]]

    """
    num_rows, num_cols = matrix.shape
    the_matrix = np.copy(matrix)
    transform_matrix = np.identity(num_rows, dtype=np.int64)
    pivot_row = 0
    pivot_cols = np.empty(min(num_rows, num_cols), dtype=np.int64)
    pivot_cols_count = 0

    for col in range(num_cols):
        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if the_matrix[pivot_row, col] != 1:
            # Find a row with a 1 in this col
            swap_row_index = pivot_row + np.argmax(the_matrix[pivot_row:num_rows, col])
            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if the_matrix[swap_row_index, col] == 1:
                # Swap rows
                # the_matrix[[swap_row_index, pivot_row]] = the_matrix[[pivot_row, swap_row_index]]
                tmp = np.copy(the_matrix[pivot_row])
                the_matrix[pivot_row] = the_matrix[swap_row_index]
                the_matrix[swap_row_index] = tmp

                # Transformation matrix update to reflect this row swap
                # transform_matrix[[swap_row_index, pivot_row]] = transform_matrix[[pivot_row, swap_row_index]]
                tmp = np.copy(transform_matrix[pivot_row])
                transform_matrix[pivot_row] = transform_matrix[swap_row_index]
                transform_matrix[swap_row_index] = tmp

        # If we have got a pivot, now let's ensure values below that pivot are zeros
        if the_matrix[pivot_row, col]:
            if not full:  
                elimination_range = np.arange(pivot_row + 1, num_rows)
            else:
                elimination_range = np.concatenate((np.arange(pivot_row), np.arange(pivot_row + 1, num_rows)))

            # Let's zero those values below the pivot by adding our current row to their row
            mask = the_matrix[elimination_range, col] == 1
            the_matrix[elimination_range[mask]] = (the_matrix[elimination_range[mask]] + the_matrix[pivot_row]) % 2
            transform_matrix[elimination_range[mask]] = (transform_matrix[elimination_range[mask]] + transform_matrix[pivot_row]) % 2
            
            pivot_cols[pivot_cols_count] = col
            pivot_cols_count += 1
            pivot_row += 1

        # Exit loop once there are no more rows to search
        if pivot_row >= num_rows:
            break

    # The rank is equal to the maximum pivot index
    matrix_rank = pivot_row
    return the_matrix, matrix_rank, transform_matrix, pivot_cols[:pivot_cols_count]

def rank(matrix):
    """
    Returns the rank of a binary matrix

    Parameters
    ----------

    matrix: numpy.ndarray
        A binary matrix in numpy.ndarray format

    Returns
    -------
    int
        The rank of the matrix
    

    Examples
    --------
    >>> H=np.array([[1,0,0],[0,1,0],[0,0,1]])
    >>> print(rank(H))
    3

    >>> H=np.array([[1,1,0],[0,1,1],[1,0,1]])
    >>> print(rank(H))
    2

    """
    return row_echelon(matrix, full=False)[1]

def nullspace(matrix):
    """
    Computes the nullspace of the matrix M. Also sometimes referred to as the kernel.

    All vectors x in the nullspace of M satisfy the following condition::

        Mx=0 \forall x \in nullspace(M)
   
    Notes
    -----
    Why does this work?

    The transformation matrix, P, transforms the matrix M into row echelon form, ReM::

        P@M=ReM=[A,0]^T,
    

    where the width of A is equal to the rank. This means the bottom n-k rows of P
    must produce a zero vector when applied to M. For a more formal definition see
    the Rank-Nullity theorem.

    Parameters
    ----------
    matrix: numpy.ndarray or scipy.sparse
        A binary matrix in numpy.ndarray or scipy.sparse format
    
    Returns
    -------
    numpy.ndarray or scipy.sparse
        A binary matrix where each row is a nullspace vector of the inputted binary
        matrix
    
    Examples
    --------
    >>> H=np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]])
    >>> print(nullspace(H))
    [[1 1 1 0 0 0 0]
     [0 1 1 1 1 0 0]
     [0 1 0 1 0 1 0]
     [0 0 1 1 0 0 1]]
    """
    transpose = np.ascontiguousarray(matrix.T, dtype=np.int64)
    m, n = transpose.shape
    _, matrix_rank, transform, _ = row_echelon(transpose, full=False)
    nspace = transform[matrix_rank:m]
    return nspace

def row_span(matrix):
    """
    Outputs the span of the row space of the matrix i.e. all linear combinations of the rows


    Parameters
    ----------
    matrix: numpy.ndarray
        The input matrix

    Returns
    -------
    numpy.ndarray
        A numpy.ndarray matrix with rows reperesenting all linear combinations of the rows of
        the inputted matrix.

    Examples
    --------
    >>> H=np.array([[1,1,0],[0,1,1],[1,0,1]])
    >>> print(row_span(H))
    [[0 0 0]
     [0 1 1]
     [1 0 1]
     [1 1 0]]
    """

    span = []
    for row in matrix:
        temp = [row]
        for element in span:
            temp.append((row + element) % 2)
        span = list(np.unique(temp + span, axis=0))
    if span:
        return np.vstack(span)
    else:
        return np.array([])


def inverse(matrix):
    """
    Computes the left inverse of a full-rank matrix.

    Notes
    -----

    The `left inverse' is computed when the number of rows in the matrix
    exceeds the matrix rank. The left inverse is defined as follows::

        Inverse(M.T@M)@M.T

    We can make a further simplification by noting that the row echelon form matrix
    with full column rank has the form::

        row_echelon_form=P@M=vstack[I,A]

    In this case the left inverse simplifies to::

        Inverse(M^T@P^T@P@M)@M^T@P^T@P=M^T@P^T@P=row_echelon_form.T@P

    Parameters
    ----------
    matrix: numpy.ndarray
        The binary matrix to be inverted in numpy.ndarray format. This matrix must either be
        square full-rank or rectangular with full-column rank.

    Returns
    -------
    numpy.ndarray
        The inverted binary matrix


    Examples
    --------

    >>> # full-rank square matrix
    >>> mat=np.array([[1,1,0],[0,1,0],[0,0,1]])
    >>> i_mat=inverse(mat)
    >>> print(i_mat@mat%2)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    >>> # full-column rank matrix
    >>> mat=np.array([[1,1,0],[0,1,0],[0,0,1],[0,1,1]])
    >>> i_mat=inverse(mat)
    >>> print(i_mat@mat%2)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    """
    m, n = matrix.shape
    row_echelon_form, matrix_rank, transform, _ = row_echelon(matrix, full=True)
    if m == n and matrix_rank == m:
        return transform

    # compute the left-inverse
    elif m > matrix_rank and n == matrix_rank:  # left inverse
        return row_echelon_form.T @ transform % 2

    else:
        raise ValueError("This matrix is not invertible. Please provide either a full-rank square\
        matrix or a rectangular matrix with full column rank.")

def row_basis(matrix):
    """
    Outputs a basis for the rows of the matrix.


    Parameters
    ----------
    matrix: numpy.ndarray or scipy.sparse
        The input matrix

    Returns
    -------
    numpy.ndarray
        A numpy.ndarray matrix where each row is a basis element.
    
    Examples
    --------

    >>> H=np.array([[1,1,0],[0,1,1],[1,0,1]])
    >>> rb=row_basis(H)
    >>> print(rb)
    [[1 1 0]
     [0 1 1]]
    """
    transpose = np.ascontiguousarray(matrix.T, dtype=np.int64)
    return matrix[row_echelon(transpose, full=False)[3]]

if __name__ == "__main__":
    H = np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]], dtype=np.int64)
    row_echelon_form, matrix_rank, transform, _ = row_echelon(H, full=False)
    print(row_echelon_form)
    ns = nullspace(H)
    print(f'k = {len(ns)}')

    