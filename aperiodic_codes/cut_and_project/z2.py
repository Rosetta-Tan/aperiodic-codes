"""
Assorted functions to work with binary vectors and matrices
Copyright: Joschka Roffe, 2021
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

def row_echelon(matrix, full=False):
    
    """
    Converts a binary matrix to row echelon form via Gaussian Elimination

    Parameters
    ----------
    matrix : numpy.ndarry or scipy.sparse
        A binary matrix in either numpy.ndarray format or scipy.sparse
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

    def _xor_vecs(vec1, vec2):
        if isinstance(vec1, csr_matrix) and isinstance(vec2, csr_matrix):
            return (vec1!=vec2).astype(int)
        else:
            return (vec1 + vec2) % 2

    num_rows, num_cols = np.shape(matrix)

    # Take copy of matrix if numpy (why?) and initialise transform matrix to identity
    if isinstance(matrix, np.ndarray):
        the_matrix = np.copy(matrix)
        transform_matrix = np.identity(num_rows).astype(int)
    elif isinstance(matrix, csr_matrix):
        the_matrix = matrix
        transform_matrix = sparse.eye(num_rows, dtype="int", format="csr")
    else:
        raise ValueError('Unrecognised matrix type')

    pivot_row = 0
    pivot_cols = []

    # print(f'debug: the_matrix: {the_matrix}')

    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(num_cols):

        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if the_matrix[pivot_row, col] != 1:

            # Find a row with a 1 in this col
            swap_row_index = pivot_row + np.argmax(the_matrix[pivot_row:num_rows, col])

            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if the_matrix[swap_row_index, col] == 1:

                # Swap rows
                the_matrix[[swap_row_index, pivot_row]] = the_matrix[[pivot_row, swap_row_index]]

                # Transformation matrix update to reflect this row swap
                transform_matrix[[swap_row_index, pivot_row]] = transform_matrix[[pivot_row, swap_row_index]]

        # If we have got a pivot, now let's ensure values below that pivot are zeros
        if the_matrix[pivot_row, col]:

            if not full:  
                elimination_range = [k for k in range(pivot_row + 1, num_rows)]
            else:
                elimination_range = [k for k in range(num_rows) if k != pivot_row]

            # Let's zero those values below the pivot by adding our current row to their row
            for j in elimination_range:

                if the_matrix[j, col] != 0 and pivot_row != j:    ### Do we need second condition?
                    # print(f'debug: the_matrix[j].__class__: {the_matrix[j].__class__}')
                    # print(f'debug: the_matrix[j]: {the_matrix[j]}')
                    # print(f'debug: the_matrix[pivot_row].__class__: {the_matrix[pivot_row].__class__}')
                    # print(f'debug: the_matrix[pivot_row]: {the_matrix[pivot_row]}')
                    the_matrix[j] = _xor_vecs(the_matrix[j], the_matrix[pivot_row])

                    # Update transformation matrix to reflect this op
                    transform_matrix[j] = _xor_vecs(transform_matrix[j], transform_matrix[pivot_row])

            pivot_row += 1
            pivot_cols.append(col)

        # Exit loop once there are no more rows to search
        if pivot_row >= num_rows:
            break

    # The rank is equal to the maximum pivot index
    matrix_rank = pivot_row
    row_esch_matrix = the_matrix

    return [row_esch_matrix, matrix_rank, transform_matrix, pivot_cols]

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
    transpose = matrix.T
    if isinstance(matrix, csr_matrix):
        transpose = transpose.tocsr()
    m, n = transpose.shape
    _, matrix_rank, transform, _ = row_echelon(transpose)
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
    row_echelon_form, matrix_rank, transform, _ = row_echelon(matrix, True)
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
    transpose = matrix.T
    if isinstance(matrix, csr_matrix):
        transpose = transpose.tocsr()
    return matrix[row_echelon(transpose)[3]]

if __name__ == "__main__":
    H = np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]])
    ns = nullspace(H)

    H_sp = csr_matrix(H)
    ns_sp = nullspace(H_sp)
    
    print(np.allclose(ns, ns_sp.toarray()))

    print(row_basis(H_sp).shape[1])