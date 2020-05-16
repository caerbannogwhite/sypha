import numpy

from pathlib import Path
from model_importer import *


def initial_point_computation(mat: numpy.array, rhs: numpy.array, obj: numpy.array) -> (numpy.array, numpy.array, numpy.array):
    """
    Compute the Mehrotra’s initial point heuristic.

    Parameters
    ----------
    mat : numpy.array

    rhs : numpy.array

    obj : numpy.array


    Returns
    -------
    x :
        Initial x vector.
    y :
        Initial y vector.
    s :
        Initial s vector.
    """

    # A.T * A can be singular: need to find a solution for this
    matTmat_inv = numpy.linalg.inv(mat.T.dot(mat))

    x = matTmat_inv.dot(mat.T.dot(rhs))
    y = mat.dot(matTmat_inv).dot(obj)
    s = rhs - mat.dot(x)
    
    delta_y = max(-1.5 * numpy.min(y), 0.0)
    delta_s = max(-1.5 * numpy.min(s), 0.0)

    y_hat = y + delta_y
    s_hat = s + delta_s

    p = y_hat.dot(s_hat)

    delta_y_hat = 0.5 * p / numpy.sum(s_hat)
    delta_s_hat = 0.5 * p / numpy.sum(y_hat)

    return x, y_hat + delta_y_hat, s_hat + delta_s_hat


def mehrotra_linopt(mat: numpy.array,
                    rhs: numpy.array,
                    obj: numpy.array,
                    eta: float,
                    k_max: int,
                    tol_p=1E-9,
                    tol_d=1E-6,
                    tol_o=1E-9):
    """
    Compute the Mehrotra’s predictor-corrector algorithm for linear optimization
    problems in the inequality form with direct solver.
    
    Parameters
    ----------

    Returns
    -------

    """

    m, n = mat.shape

    # get the initial point
    x, y, s = initial_point_computation(mat, rhs, obj)

    # residuals
    r_d = obj - mat.T.dot(rhs)
    r_p = rhs - mat.dot(x) # TODO + z??
    r_c = -numpy.diag(s).dot(numpy.diag(y)).dot(numpy.ones(m))

    # duality measure
    mu = s.dot(y) / float(m)

    iterations = 0

    while (
        (iterations < k_max) and
        ((numpy.linalg.norm(r_p) / (1 + numpy.linalg.norm(rhs))) < tol_p) and
        ((numpy.linalg.norm(r_d) / (1 + numpy.linalg.norm(obj))) < tol_d) and
        ((mu / (1 + obj.dot(x))) < tol_o)
    ):

        Y = numpy.diag(y)
        Y_inv = numpy.linalg.inv(Y)
        S = numpy.diag(s)
        D = numpy.linalg.inv(S).dot(Y)
        AD = mat.T.dot(D)

        L = numpy.linalg.cholesky(AD.dot(mat))

        # affine step
        r_aff = AD.dot(r_p) - (r_d + Y_inv.dot(r_c))
        delta_x_aff = None
        delta_y_aff = D.dot(r_d - mat.dot(delta_x_aff))
        delta_s_aff = Y_inv.dot(r_c - S.dot(delta_y_aff))

        # affine step length
        alpha_p = min(1.0, min([-xi / delta_xi for xi, delta_xi in zip(x, delta_x_aff) if delta_xi < 0.0]))
        alpha_d = min(1.0, min([-si / delta_si for si, delta_si in zip(s, delta_s_aff) if delta_si < 0.0]))
        mu_aff = (s + alpha_p.dot(delta_s_aff)).dot(y + alpha_d.dot(delta_y_aff)) / float(m)

        # corrector step
        sigma = (mu_aff / mu) ** 3

        h = - numpy.diag(delta_s_aff).dot(numpy.diag(delta_y_aff)).dot(numpy.ones(m)) + sigma * mu
        delta_x_cor = None
        delta_y_cor = D.dot(-mat.dot(delta_x_cor))
        delta_s_cor = Y_inv.dot(h - S.dot(delta_y_cor))

        # predictor step
        delta_x = delta_x_aff + delta_x_aff
        delta_y = delta_y_aff + delta_y_aff
        delta_s = delta_s_aff + delta_s_aff

        # TODO alpha ???

        x += eta * alpha_p * delta_x
        y += eta * alpha_d * delta_y
        s += eta * alpha_d * delta_s
        
        # residuals
        r_d = obj - mat.T.dot(rhs)
        r_p = rhs - mat.dot(x) # TODO + z???
        r_c = -numpy.diag(s).dot(numpy.diag(y)).dot(numpy.ones(m))

        # update
        mu = s.dot(y) / float(m)
        iterations += 1

    return x, y, s



path = Path(r"C:\Users\IP 520S-14IKB 96IX\coding\sypha\data\scp_demo13.txt")

mat, rhs, obj = sc_model_reader(path)
#sol = mehrotra_linopt(mat, rhs, obj, 0.99995, 1000)

x, y, s = initial_point_computation(mat, rhs, obj)

#print("obj val =", sol.dot(obj))

