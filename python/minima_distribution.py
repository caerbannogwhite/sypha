
import numpy as np

#function [X ,Y , Hdata ] = mdopt (TF ,iD , opt )

def mdopt(tf, lb, ub, S0=None, S=None, S_MAX=None, R=0.8, tol=1E-3):
    """
    MDOPT finds a global minima of a continuous f on a compact set.
    MDOPT attempts to solve problems of the form:
    
    min TF(X) subject to: lb <= X <= ub (bounds)
    X iD {3}(X) > 0 (non linear constraints)
    
    where
    S0 is the number of samples at the beginning,
    S is the number of samples per update,
    R is the recognition accuracy (0.66 < R < 1) as a termination
    criterion (TC) for the inner loop,
    S_MAX is the max number of samples for the inner loop,
    tol is the tolerance as a TC for the outer loop.
    
    Default params
    --------------

    S0 = 20 * d
    S = 10 * d
    R = 0.8
    S_MAX = 100 * d
    tol = 1e-3
    
    References
    ----------

    X Luo, "Minima distribution for global optimization", arXiv:1812.03457
    Xiaopeng Luo, 2/13/2019
    """

    print("mdopt start")
    
    # parameters setting
    n = len(lb)
    if not S0:
        S0 = 20 * n

    if not s:
        S = 10 * n

    if not S_MAX:
        S_MAX = 100 * n

    # initialization
    k = 0
    iterations = 1
    iMk = genmk(1, [], [], S0)
    iDk = []
    nss, X = splD(1, 0, 0, iMk, lb, ub) # samples
    Y = tf(X)
    xb, yb, xc, yc = updbest(X, Y, np.zeros(n), np.inf, np.inf, np.inf)
    #rec, HX, HY = mdrecord([], 0, iterations, 0, X, Y, yb)
    
    while (yc > tol):
        
        k = np.max(k, 1.0 / Y.std()) # update k 
        EY = Y.mean()
        T = np.exp(-k * (Y - EY))
        MT = T.mean()
        IF = np.isinf(T)
        
        if IF.any():
            T[IF] = 1
            T[np.logical_not(IF)] = 0
            MT = T.mean()

        # update the current best and generate m^k for sampling
        xb, yb, xc, yc = updbest(X, Y, xb, yb, xc, yc)
        iMk = genmk(0, X, T, S)
        
        # update D^k and samples if iter > 1
        if iterations > 1:
            iDk = { MT ;k ; EY ; rbf }
            I = T >= MT
            [X ,Y] = deal (X[I, :] ,Y[I] )

        # inner loop
        RA = 0 
        inneriter = 0
        
        while (RA < R) and (X.shape(0) < S_MAX) and (yc > tol):
            
            inneriter = inneriter + 1
            
            # update the approximation
            rbf = adrbf(X ,Y)
            
            # sampling in D ^ k
            nss, Xn = splD(0, inneriter == 1, nss + 1, iMk, iD, iDk)
            eYn = rbfp(Xn, rbf)
            
            # taking minima of the approximation
            iMk0 = {100* iMk {1}; iMk {2}; iMk {3}/2}
            _, Xn2 = splD(0, 1, 1, iMk0, lb, ub, iDk)
            eYn2 = rbfp(Xn2, rbf)
            [~ , I2min ] = min ( eYn2 ,[] , ’ omitnan ’)
            [~ , Imax ] = max (eYn ,[] , ’ omitnan ’)
            Xn[Imax, :] = Xn2[I2min, :]
            eYn[Imax] = eYn2[I2min]
            
            # call target function
            Yn = tf(Xn)
            xb, yb, xc, yc = updbest(Xn, Yn, xb, yb, xc, yc)
            
            # termination criteria for inner loop
            eTn = np.exp(-k * (eYn - EY))
            Tn = np.exp(-k * (Yn - EY))
            T = np.vstack((T, Tn))
            MT = T.mean()
            ET = np.abs(eTn - Tn).mean()
            
            # recognition accuracy (RA)
            if k < 10:
                RA = sum (( eTn > MT ) == ( Tn > MT ) )/ length ( Tn )
            else:
                It = Tn > MT
                V1 = ( eTn ( It ) -ET ) > MT
                V2 = ( eTn (~ It ) + ET ) <= MT ;
                RA = (sum ( V1 ) + sum ( V2 )/ length ( Tn ))
        
            # update , record , and plot
            X = np.vstack((X, Xn))
            Y = np.vstack((Y, Yn))
            
            #rec, HX, HY = mdrecord(rec, iterations, inneriter, k, Xn, Yn, yb, HX, HY)
            #mdplot(rec)

        # update iter
        #fprintf ( ’ - iteration %d with the best obj = %d ;\ n ’,iter , yb )
        iterations = iterations + 1

    X = xb
    Y = yb
    #Hdata = { HX ; HY ; rec }
    Hdata = None
    return X, Y, Hdata


def splD(st, fp, nss, iMk, lb, up, constraints=[], iDk=None):
    # SPLD generates samples Xn by sampling Mk in Dk
    if st == 1: # for the beginning
        if nss == 0:
            nss = np.random.randint(1, 101)

        # https://uk.mathworks.com/help/stats/haltonset.html
        # https://uk.mathworks.com/help/stats/haltonset.net.html
        #ax = net(haltonset(d , ’Skip’, nss + 1), iMk[0])
        ax = np.random.rand(iMk[0], n)

        # https://uk.mathworks.com/help/matlab/ref/bsxfun.html
        #ax = bsxfun(@plus, bsxfun(@times, ax, (UB - LB)’), LB’)
        ax = ax * (ub - lb) - lb

        # remember iD{3} soubld be the set of the constraints...  c(x) > 0
        #if len(constraints) > 0:
        #     ax = ax ( iD {3 ,1}( ax ) ,:)

        Xn = ax
        nss = nss + iMk[0]
    else:
        [beta ,Xi , sig ] = deal ( iMk {1} , iMk {2} , iMk {3})
        
        d = size (Xi ,2)
        Xn = []
        i = 1
        while (i <= length (beta)):
            if (fp == 1) and (i == 1):
                nx = beta[i] - 1
                X0 = Xi[i, :]
            else:
                nx = beta[i]
                X0 = []
        
        counter = 0
        iDk0 = iDk
        while (nx > 0):
            axi = net(haltonset(d, 'Skip', nss + 1), nx)
            nss = nss + nx
            axi = norminv(axi, Xi[i, :], sig(i))
            axi = inD(axi, iD, iDk0)
            if isempty(axi):
                counter = counter + 1

            """
            if (counter > 10) and (not isempty ( iDk )):
                counter = 0
                iDk0{1} = 0.9 * iDk0 {1}
                sig (i) = 0.9 * sig (i)
                if (iDk0 {1} < 1E-6) or (sig (i ) < 1E-6):
                    axi = net ( haltonset (d , 'Skip' , nss +1) ,nx )
                    nss = nss + nx
                    axi = norminv ( axi , Xi (i ,:) , sig (i ))

                nx = nx - axi.shape[0]
                X0 = np.vstack((X0, axi))
            """
            X0 = X0[1:beta(i), :]
            Xn = np.vstack((Xn, X0))
            i = i + 1

    return nss, Xn


def inD(X, iD, iDk):
    if isempty (X):
        return

    L = iD {1}
    U = iD {2}
    d = size (X ,2)
    if length ( iD ) > 2:
        iDn = iD {3}

    if ~ isempty ( iDk ):
        [ TM ,a , EY , rbf ] = deal ( iDk {1} , iDk {2} , iDk {3} , iDk {4})

    X = X (sum (bsxfun ( @ge ,X ,L ’) ,2) == d ,:) ;
    X = X (sum (bsxfun ( @le ,X ,U ’) ,2) == d ,:) ;
    
    if length ( iD ) > 2:
        X = X ( iDn (X ) ,:)

    if ~ isempty ( iDk ):
        Y = rbfp (X , rbf )
        T = np.exp(-a * (Y - EY))
        X = X (T >= TM ,:)
    
    return X


def updbest(X, Y, xb, yb, xchg, ychg):
    I = Y.argmin(axis=0)
    if Y[I[0]] < yb:
        xchg = np.linalg.norm(X[I[0], :] - xb)
        ychg = yb - Y[I[0]]
        xb = X[I[0], :]
        yb = Y[I[0]]

    return xb, yb, xchg, ychg


def genmk(st, X, T, s):
    if st == 1:  # for the beginning
        return [s]

    else:
        Xc = (T / T.sum()).dot(X)
        Dc = X.std(axis=0, ddof=1).mean()
        #[~ , I] = max(T)
        I = T.argmax(axis=0)
        Xc2 = X[I, :]
        N2 = 1
        return [s - N2, N2], [Xc, Xc2], [Dc, Dc / 2.0]


"""
function rbf = adrbf (X ,Y)
rbf = fitrgp (X ,Y , ’ KernelFunction ’,’ squaredexponential ’ ,...
’ FitMethod ’ ,’ exact ’ ,’ Standardize ’ ,1) ;
end

function y = rbfp (x , rbf )
y = predict ( rbf ,x) ;
end
"""