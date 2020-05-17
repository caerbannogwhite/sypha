import numpy

def initial_point_computation(mat: numpy.array,
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


def mehrotra_linopt(mat: numpy.array,
                    rhs: numpy.array,
                    obj: numpy.array,
                    eta: float,
                    k_max: int,
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

    m, n = mat.shape

    # get the initial point
    x, y, s = initial_point_computation(mat, rhs, obj)

    # residuals
    # r_b, r_c equation 14.7, page 395(414) Numerical Optimization
    r_b = mat.dot(x) - rhs
    r_c = mat.T.dot(y) + s - obj

    # duality measure, defined at page 395(414) Numerical Optimization
    mu = x.dot(s) / float(n)

    iterations = 0

    while (
        (iterations < k_max) and
        #((numpy.linalg.norm(r_p) / (1 + numpy.linalg.norm(rhs))) < tol_p) and
        #((numpy.linalg.norm(r_d) / (1 + numpy.linalg.norm(obj))) < tol_d) and
        #((mu / (1 + obj.dot(x))) < tol_o)
        (mu > 1E-16)
    ):

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

        # affine step length, definition 14.32 at page 408(427)
        alpha_max_p = min([-xi / delta_xi for xi, delta_xi in zip(x, delta_x_aff) if delta_xi < 0.0])
        alpha_max_d = min([-si / delta_si for si, delta_si in zip(s, delta_s_aff) if delta_si < 0.0])
        alpha_aff_p = min(1.0, alpha_max_p)
        alpha_aff_d = min(1.0, alpha_max_d)
        mu_aff = (x + alpha_aff_p * delta_x_aff).dot(s + alpha_aff_d * delta_s_aff) / float(m)

        # corrector step or centering parameter
        sigma = (mu_aff / mu) ** 3

        DELTA_X_aff = numpy.diag(delta_x_aff)
        DELTA_S_aff = numpy.diag(delta_s_aff)
        r_xs += DELTA_X_aff.dot(DELTA_S_aff).dot(numpy.ones(n)) - sigma * mu

        delta_y = numpy.linalg.solve(ADA, -r_b - mat.dot(D).dot(r_c) + mat.dot(S_inv.dot(r_xs)))
        delta_s = -r_c - mat.T.dot(delta_y)
        delta_x = -S_inv.dot(r_xs) - D.dot(delta_s)

        alpha_p = min(1.0, eta * alpha_max_p)
        alpha_d = min(1.0, eta * alpha_max_d)

        x += alpha_p * delta_x
        y += alpha_d * delta_y
        s += alpha_d * delta_s
        
        #print(mu, mu_aff, sigma)

        # update
        r_b = mat.dot(x) - rhs
        r_c = mat.T.dot(y) + s - obj

        mu = x.dot(s) / float(n)

        iterations += 1

    return x, y, s, iterations

