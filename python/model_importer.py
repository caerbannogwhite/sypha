import numpy
import scipy.sparse

from scipy.sparse import csr_matrix
from pathlib import Path

def sc_dense_model_reader(input_file_path: Path) -> (numpy.array, numpy.array, numpy.array):
    """
    Given a file path, read the Set Covering model contained and
    return the matrix and the cost vector.

    A Set Covering model file is defined as:
    m n
    c_0 c_1 ...
    ... c_n-1


    Parameters
    ----------
    input_file_path : Path

    Returns
    -------
    mat :
        The matrix of the model.
    rhs :
        The right hand side vector.
    obj :
        The cost vector of the model.
    """

    with open(input_file_path, "r") as file_handler:
        num_rows, num_cols = [int(i) for i in file_handler.readline().split()]
        
        buffer = list()

        # read the objective vector
        while len(buffer) < num_cols:
            buffer += [int(i) for i in file_handler.readline().split()]

        obj = numpy.array(buffer)
        mat = numpy.zeros((num_rows, num_cols))

        for i in range(num_rows):
            non_zero = int(file_handler.readline().strip())
            buffer = list()

            while len(buffer) < non_zero:
                buffer += [int(i) - 1 for i in file_handler.readline().split()]

            mat[i, buffer] = 1.0

    return mat, numpy.ones(num_rows), obj


def sc_dense_to_standard_form(mat, rhs, obj):

    m, n = mat.shape

    return (numpy.hstack((mat, -numpy.eye(m))),
           rhs,
           numpy.hstack((obj, numpy.zeros(m))))


def sc_sparse_model_reader(input_file_path: Path) -> (csr_matrix, numpy.array, numpy.array):
    """
    Given a file path, read the Set Covering model contained and
    return the matrix (in sparse CSR form) and the cost vector.

    A Set Covering model file is defined as:
    m n
    c_0 c_1 ...
    ... c_n-1


    Parameters
    ----------
    input_file_path : Path

    Returns
    -------
    mat :
        The matrix of the model in sparse CSR form.
    rhs :
        The right hand side vector.
    obj :
        The cost vector of the model.
    """

    with open(input_file_path, "r") as file_handler:
        num_rows, num_cols = [int(i) for i in file_handler.readline().split()]
        
        buffer = list()

        # read the objective vector
        while len(buffer) < num_cols:
            buffer += [int(i) for i in file_handler.readline().split()]

        obj = numpy.array(buffer)
        indices = list()
        indptr = [0]

        for i in range(num_rows):
            non_zero_num = int(file_handler.readline().strip())
            indptr.append(indptr[-1] + non_zero_num)
            buffer = list()

            while len(buffer) < non_zero_num:
                buffer += [int(i) - 1 for i in file_handler.readline().split()]
            
            indices += buffer

    mat = csr_matrix((
        numpy.ones(len(indices)),
        numpy.array(indices),
        numpy.array(indptr)),
        shape=(num_rows,num_cols
        ))

    return mat, numpy.ones(num_rows), obj


def sc_sparse_to_standard_form(mat, rhs, obj):

    m, n = mat.shape

    return (scipy.sparse.hstack((mat, -scipy.sparse.eye(m))),
           rhs,
           numpy.hstack((obj, numpy.zeros(m))))