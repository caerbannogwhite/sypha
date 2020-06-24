import numpy
import scipy.sparse
import warnings

from utils import print_mat, print_vec

from scipy.sparse import csr_matrix
from scipy.sparse import linalg as splinalg
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter('ignore', SparseEfficiencyWarning)

def initial_point_computation_dense(mat: numpy.array,
                              rhs: numpy.array,
                              obj: numpy.array
                              ) -> (numpy.array, numpy.array, numpy.array):
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

    # A * A.T can be singular: need to find a solution for this
    AAT_inv = numpy.linalg.inv(mat.dot(mat.T))

    x = mat.T.dot(AAT_inv).dot(rhs)
    y = AAT_inv.dot(mat).dot(obj)
    s = obj - mat.T.dot(y)

    delta_x = max(-1.5 * numpy.min(x), 0.0)
    delta_s = max(-1.5 * numpy.min(s), 0.0)

    x_hat = x + delta_x
    s_hat = s + delta_s

    p = x_hat.dot(s_hat)
    
    delta_x_hat = 0.5 * p / numpy.sum(s_hat)
    delta_s_hat = 0.5 * p / numpy.sum(x_hat)
    
    return x_hat + delta_x_hat, y, s_hat + delta_s_hat


def mehrotra_linopt_dense(mat: numpy.array,
                    rhs: numpy.array,
                    obj: numpy.array,
                    eta=0.9,
                    k_max=1000,
                    tol_p=1E-9,
                    tol_d=1E-6,
                    tol_o=1E-2):
    """
    Compute the Mehrotra’s predictor-corrector algorithm for linear optimization
    problems in the inequality form with direct solver.
    
    Parameters
    ----------

    Returns
    -------

    """

    log = 0
    m, n = mat.shape

    # get the initial point
    x, y, s = initial_point_computation_dense(mat, rhs, obj)

    # residuals
    # r_b, r_c equation 14.7, page 395(414) Numerical Optimization
    r_b = mat.dot(x) - rhs
    r_c = mat.T.dot(y) + s - obj

    # duality measure, defined at page 395(414) Numerical Optimization
    mu = x.dot(s) / float(n)

    iterations = 0

    if log > 2:
        print("\n")
        print(f"\t{'mu':10s} | {'mu aff':10s} | {'sigma':10s} | {'alpha p':10s} | {'alpha d':10s} | {'upp':10s} | {'low':10s}")
        print(f"\t{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}")

    while (
        (iterations < k_max) and
        #((numpy.linalg.norm(r_p) / (1 + numpy.linalg.norm(rhs))) < tol_p) and
        #((numpy.linalg.norm(r_d) / (1 + numpy.linalg.norm(obj))) < tol_d) and
        #((mu / (1 + obj.dot(x))) < tol_o)
        (mu > 1E-10)
    ):

        #print((x > 0.0).all(), (s > 0.0).all(), mu)

        X = numpy.diag(x)
        S = numpy.diag(s)
        r_xs = X.dot(S).dot(numpy.ones(n))

        S_inv = numpy.linalg.inv(S)
        D = X.dot(S_inv)
        ADA = mat.dot(D.dot(mat.T))
        
        # affine step
        delta_y_aff = numpy.linalg.solve(ADA, -r_b - mat.dot(D).dot(r_c) + mat.dot(S_inv.dot(r_xs)))
        delta_s_aff = -r_c - mat.T.dot(delta_y_aff)
        delta_x_aff = -S_inv.dot(r_xs) - D.dot(delta_s_aff)

        row_1 = numpy.hstack((numpy.zeros((n, n)), mat.T, numpy.eye(n, n)))
        row_2 = numpy.hstack((mat, numpy.zeros((m, m)), numpy.zeros((m, n))))
        row_3 = numpy.hstack((S, numpy.zeros((n, m)), X))
        A = numpy.vstack((row_1, row_2, row_3))
        b = numpy.hstack((-r_c, -r_b, -r_xs))
        #sol = numpy.linalg.solve(A, b)
        #delta_x_aff = sol[:n]
        #delta_s_aff = sol[n+m:]

        print("PRE AFFINE SYSTEM")
        print(A)

        print(f"\n\niter: {iterations}")
        print(f"X:\n{x}")
        print(f"Y:\n{y}")
        print(f"S:\n{s}")
        print(f"delta X:\n{delta_x_aff}")
        print(f"delta S:\n{delta_s_aff}")
        print(f"rhs:\n{b}")

        # affine step length, definition 14.32 at page 408(427)
        alpha_max_p = min([-xi / delta_xi for xi, delta_xi in zip(x, delta_x_aff) if delta_xi < 0.0])
        alpha_max_d = min([-si / delta_si for si, delta_si in zip(s, delta_s_aff) if delta_si < 0.0])
        alpha_aff_p = min(1.0, alpha_max_p)
        alpha_aff_d = min(1.0, alpha_max_d)

        mu_aff = (x + alpha_aff_p * delta_x_aff).dot(s + alpha_aff_d * delta_s_aff) / float(n)

        # corrector step or centering parameter
        sigma = (mu_aff / mu) ** 3

        print(f"\n\niter: {iterations}")
        print(f"sigma: {sigma}, mu aff: {mu_aff}\n")
        print(f"buff X:\n{x + alpha_aff_p * delta_x_aff}")
        print(f"buff S:\n{s + alpha_aff_d * delta_s_aff}")

        DELTA_X_aff = numpy.diag(delta_x_aff)
        DELTA_S_aff = numpy.diag(delta_s_aff)
        r_xs = r_xs + DELTA_X_aff.dot(DELTA_S_aff).dot(numpy.ones(n)) - sigma * mu

        delta_y = numpy.linalg.solve(ADA, -r_b - mat.dot(D).dot(r_c) + mat.dot(S_inv.dot(r_xs)))
        delta_s = -r_c - mat.T.dot(delta_y)
        delta_x = -S_inv.dot(r_xs) - D.dot(delta_s)

        b = numpy.hstack((-r_c, -r_b, -r_xs))
        #sol = numpy.linalg.solve(A, b)
        #delta_x = sol[:n]
        #delta_y = sol[n:n+m]
        #delta_s = sol[n+m:]

        print(f"\n\n{iterations:4d}) AFTER CORRECTION SYSTEM")
        print(f"X:\n{x}")
        print(f"Y:\n{y}")
        print(f"S:\n{s}")
        print(f"delta X:\n{delta_x}")
        print(f"delta S:\n{delta_s}")
        print(f"rhs:\n{b}")

        alpha_max_p = min([-xi / delta_xi for xi, delta_xi in zip(x, delta_x) if delta_xi < 0.0])
        alpha_max_d = min([-si / delta_si for si, delta_si in zip(s, delta_s) if delta_si < 0.0])
        alpha_p = min(1.0, eta * alpha_max_p)
        alpha_d = min(1.0, eta * alpha_max_d)

        x += alpha_p * delta_x
        y += alpha_d * delta_y
        s += alpha_d * delta_s
        
        if log > 2:
            upp = x[:n-m].dot(obj[:n-m])
            low = y.dot(rhs)
            print(f"\t{mu:10f} | {mu_aff:10f} | {sigma:10f} | {alpha_p:10f} | {alpha_d:10f} | {upp:10f} | {low:10f}")

        # update
        r_b = (1.0 - alpha_p) * r_b
        r_c = (1.0 - alpha_d) * r_c

        mu = x.dot(s) / float(n)

        print(f"\n\n{iterations:4d}) LOOP END")
        print(f"mu: {mu}, al prim: {alpha_p}, al dual: {alpha_d}")
        print(f"X:\n{x}")
        print(f"Y:\n{y}")
        print(f"S:\n{s}")
        print(f"delta X:\n{delta_x}")
        print(f"delta S:\n{delta_s}")
        print(f"rhs:\n{b}")

        iterations += 1

    return x, y, s, iterations


def mehrotra_linopt_dense_test(mat: numpy.array,
                               rhs: numpy.array,
                               obj: numpy.array,
                               eta=0.9,
                               k_max=1000,
                               tol_p=1E-9,
                               tol_d=1E-6,
                               tol_o=1E-2):
    """
    Compute the Mehrotra’s predictor-corrector algorithm for linear optimization
    problems in the inequality form with direct solver.
    
    Parameters
    ----------

    Returns
    -------

    """

    log = 0
    m, n = mat.shape

    # get the initial point
    x, y, s = initial_point_computation_dense(mat, rhs, obj)

    # residuals
    # r_b, r_c equation 14.7, page 395(414) Numerical Optimization
    r_b = mat.dot(x) - rhs
    r_c = mat.T.dot(y) + s - obj

    # duality measure, defined at page 395(414) Numerical Optimization
    mu = x.dot(s) / float(n)

    iterations = 0

    if log > 2:
        print("\n")
        print(f"\t{'mu':10s} | {'mu aff':10s} | {'sigma':10s} | {'alpha p':10s} | {'alpha d':10s} | {'upp':10s} | {'low':10s}")
        print(f"\t{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}")

    while (
        (iterations < k_max) and
        #((numpy.linalg.norm(r_p) / (1 + numpy.linalg.norm(rhs))) < tol_p) and
        #((numpy.linalg.norm(r_d) / (1 + numpy.linalg.norm(obj))) < tol_d) and
        #((mu / (1 + obj.dot(x))) < tol_o)
        (mu > 1E-10)
    ):

        X = numpy.diag(x)
        S = numpy.diag(s)
        r_xs = X.dot(S).dot(numpy.ones(n))

        row_1 = numpy.hstack((numpy.zeros((n, n)), mat.T, numpy.eye(n, n)))
        row_2 = numpy.hstack((mat, numpy.zeros((m, m)), numpy.zeros((m, n))))
        row_3 = numpy.hstack((S, numpy.zeros((n, m)), X))
        A = numpy.vstack((row_1, row_2, row_3))
        b = numpy.hstack((-r_c, -r_b, -r_xs))
        sol = numpy.linalg.solve(A, b)
        delta_x_aff = sol[:n]
        delta_s_aff = sol[n+m:]

        print(f"\n{iterations:4d}) AFTER AFFINE SYSTEM")
        print_mat(A)
        print(f"sol:")
        print_vec(sol)
        print(f"rhs:")
        print_vec(b)

        # affine step length, definition 14.32 at page 408(427)
        alpha_max_p = min(
            [-xi / delta_xi for xi, delta_xi in zip(x, delta_x_aff) if delta_xi < 0.0])
        alpha_max_d = min(
            [-si / delta_si for si, delta_si in zip(s, delta_s_aff) if delta_si < 0.0])
        alpha_aff_p = min(1.0, alpha_max_p)
        alpha_aff_d = min(1.0, alpha_max_d)

        mu_aff = (x + alpha_aff_p * delta_x_aff).dot(s +
                                                     alpha_aff_d * delta_s_aff) / float(n)

        # corrector step or centering parameter
        sigma = (mu_aff / mu) ** 3

        # print(f"\n\niter: {iterations}")
        # print(f"sigma: {sigma}, mu aff: {mu_aff}\n")
        # print(f"buff X:\n{x + alpha_aff_p * delta_x_aff}")
        # print(f"buff S:\n{s + alpha_aff_d * delta_s_aff}")

        DELTA_X_aff = numpy.diag(delta_x_aff)
        DELTA_S_aff = numpy.diag(delta_s_aff)
        r_xs = r_xs + \
            DELTA_X_aff.dot(DELTA_S_aff).dot(numpy.ones(n)) - sigma * mu

        # delta_y = numpy.linalg.solve(
        #     ADA, -r_b - mat.dot(D).dot(r_c) + mat.dot(S_inv.dot(r_xs)))
        # delta_s = -r_c - mat.T.dot(delta_y)
        # delta_x = -S_inv.dot(r_xs) - D.dot(delta_s)

        b = numpy.hstack((-r_c, -r_b, -r_xs))
        sol = numpy.linalg.solve(A, b)
        delta_x = sol[:n]
        delta_y = sol[n:n+m]
        delta_s = sol[n+m:]

        #print(f"\n{iterations:4d}) AFTER CORRECTION SYSTEM")
        #print(f"sol:")
        #print_vec(sol)
        #print(f"rhs:")
        #print_vec(b)

        alpha_max_p = min(
            [-xi / delta_xi for xi, delta_xi in zip(x, delta_x) if delta_xi < 0.0])
        alpha_max_d = min(
            [-si / delta_si for si, delta_si in zip(s, delta_s) if delta_si < 0.0])
        alpha_p = min(1.0, eta * alpha_max_p)
        alpha_d = min(1.0, eta * alpha_max_d)

        x += alpha_p * delta_x
        y += alpha_d * delta_y
        s += alpha_d * delta_s

        if log > 2:
            upp = x[:n-m].dot(obj[:n-m])
            low = y.dot(rhs)
            print(
                f"\t{mu:10f} | {mu_aff:10f} | {sigma:10f} | {alpha_p:10f} | {alpha_d:10f} | {upp:10f} | {low:10f}")

        # update
        r_b = (1.0 - alpha_p) * r_b
        r_c = (1.0 - alpha_d) * r_c

        mu = x.dot(s) / float(n)

        #print(f"\n{iterations:4d}) UPDATE STEP")
        #print(f"mu: {mu:8.6f}, al prim: {alpha_p:8.6f}, al max prim: {alpha_max_p:8.6f}, al dual: {alpha_d:8.6f}, al max dual: {alpha_max_d:8.6f}")


        # print(f"\n\n{iterations:4d}) LOOP END")
        # print(f"mu: {mu}, al prim: {alpha_p}, al dual: {alpha_d}")
        # print(f"X:\n{x}")
        # print(f"Y:\n{y}")
        # print(f"S:\n{s}")
        # print(f"delta X:\n{delta_x}")
        # print(f"delta S:\n{delta_s}")
        # print(f"rhs:\n{b}")

        iterations += 1

    return x, y, s, iterations



def initial_point_computation_sparse(mat: csr_matrix,
                              rhs: numpy.array,
                              obj: numpy.array
                              ) -> (numpy.array, numpy.array, numpy.array):
    """
    Compute the Mehrotra’s initial point heuristic.

    Parameters
    ----------
    mat : csr_matrix

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

    # A * A.T can be singular: need to find a solution for this
    AAT_inv = splinalg.inv(mat.dot(mat.T))

    x = mat.T.dot(AAT_inv).dot(rhs)
    y = AAT_inv.dot(mat).dot(obj)
    s = obj - mat.T.dot(y)

    delta_x = max(-1.5 * numpy.min(x), 0.0)
    delta_s = max(-1.5 * numpy.min(s), 0.0)

    x_hat = x + delta_x
    s_hat = s + delta_s

    p = x_hat.dot(s_hat)

    delta_x_hat = 0.5 * p / numpy.sum(s_hat)
    delta_s_hat = 0.5 * p / numpy.sum(x_hat)

    return x_hat + delta_x_hat, y, s_hat + delta_s_hat


def mehrotra_linopt_sparse(mat: csr_matrix,
                    rhs: numpy.array,
                    obj: numpy.array,
                    eta=0.9,
                    k_max=1000,
                    tol_p=1E-9,
                    tol_d=1E-6,
                    tol_o=1E-2):
    """
    Compute the Mehrotra’s predictor-corrector algorithm for linear optimization
    problems in the inequality form with direct solver.
    
    Parameters
    ----------

    Returns
    -------

    """

    log = 0
    m, n = mat.shape

    # get the initial point
    x, y, s = initial_point_computation_sparse(mat, rhs, obj)

    # residuals
    # r_b, r_c equation 14.7, page 395(414) Numerical Optimization
    r_b = mat.dot(x) - rhs
    r_c = mat.T.dot(y) + s - obj

    # duality measure, defined at page 395(414) Numerical Optimization
    mu = x.dot(s) / float(n)

    iterations = 0

    if log > 2:
        print("\n")
        print(f"\t{'mu':10s} | {'mu aff':10s} | {'sigma':10s} | {'alpha p':10s} | {'alpha d':10s} | {'upp':10s} | {'low':10s}")
        print(f"\t{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}-|-{'-'*10:10s}")

    while (
        (iterations < k_max) and
        #((numpy.linalg.norm(r_p) / (1 + numpy.linalg.norm(rhs))) < tol_p) and
        #((numpy.linalg.norm(r_d) / (1 + numpy.linalg.norm(obj))) < tol_d) and
        #((mu / (1 + obj.dot(x))) < tol_o)
        (mu > 1E-10)
    ):

        #print((x > 0.0).all(), (s > 0.0).all(), mu)

        X = scipy.sparse.diags(x)
        S = scipy.sparse.diags(s)
        r_xs = X.dot(S).dot(numpy.ones(n))

        S_inv = splinalg.inv(S)
        D = X.dot(S_inv)
        ADA = mat.dot(D.dot(mat.T))
        # affine step
        delta_y_aff = splinalg.spsolve(ADA, -r_b - mat.dot(D).dot(r_c) + mat.dot(S_inv.dot(r_xs)))
        delta_s_aff = -r_c - mat.T.dot(delta_y_aff)
        delta_x_aff = -S_inv.dot(r_xs) - D.dot(delta_s_aff)

        #row_1 = numpy.hstack((numpy.zeros((n, n)), mat.T, numpy.eye(n, n)))
        #row_2 = numpy.hstack((mat, numpy.zeros((m, m)), numpy.zeros((m, n))))
        #row_3 = numpy.hstack((S, numpy.zeros((n, m)), X))
        #A = numpy.vstack((row_1, row_2, row_3))
        #b = numpy.hstack((-r_c, -r_b, -r_xs))
        #sol = numpy.linalg.solve(A, b)
        #delta_x_aff = sol[:n]
        #delta_s_aff = sol[n+m:]

        # affine step length, definition 14.32 at page 408(427)
        alpha_max_p = min([-xi / delta_xi for xi, delta_xi in zip(x, delta_x_aff) if delta_xi < 0.0])
        alpha_max_d = min([-si / delta_si for si, delta_si in zip(s, delta_s_aff) if delta_si < 0.0])
        alpha_aff_p = min(1.0, alpha_max_p)
        alpha_aff_d = min(1.0, alpha_max_d)

        mu_aff = (x + alpha_aff_p * delta_x_aff).dot(s + alpha_aff_d * delta_s_aff) / float(n)

        # corrector step or centering parameter
        sigma = (mu_aff / mu) ** 3

        DELTA_X_aff = scipy.sparse.diags(delta_x_aff)
        DELTA_S_aff = scipy.sparse.diags(delta_s_aff)
        r_xs = r_xs + DELTA_X_aff.dot(DELTA_S_aff).dot(numpy.ones(n)) - sigma * mu

        delta_y = splinalg.spsolve(ADA, -r_b - mat.dot(D).dot(r_c) + mat.dot(S_inv.dot(r_xs)))
        delta_s = -r_c - mat.T.dot(delta_y)
        delta_x = -S_inv.dot(r_xs) - D.dot(delta_s)

        #b = numpy.hstack((-r_c, -r_b, -r_xs))
        #sol = numpy.linalg.solve(A, b)
        #delta_x = sol[:n]
        #delta_y = sol[n:n+m]
        #delta_s = sol[n+m:]

        alpha_max_p = min([-xi / delta_xi for xi, delta_xi in zip(x, delta_x) if delta_xi < 0.0])
        alpha_max_d = min([-si / delta_si for si, delta_si in zip(s, delta_s) if delta_si < 0.0])
        alpha_p = min(1.0, eta * alpha_max_p)
        alpha_d = min(1.0, eta * alpha_max_d)

        x += alpha_p * delta_x
        y += alpha_d * delta_y
        s += alpha_d * delta_s
        
        if log > 2:
            upp = x[:n-m].dot(obj[:n-m])
            low = y.dot(rhs)
            print(f"\t{mu:10f} | {mu_aff:10f} | {sigma:10f} | {alpha_p:10f} | {alpha_d:10f} | {upp:10f} | {low:10f}")

        # update
        r_b = (1.0 - alpha_p) * r_b
        r_c = (1.0 - alpha_d) * r_c

        mu = x.dot(s) / float(n)

        iterations += 1

    return x, y, s, iterations

