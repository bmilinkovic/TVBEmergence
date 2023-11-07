import numpy as np
from scipy.linalg import qr, lu, ordqz, logm, solve_discrete_lyapunov, cholesky, inv
import sys
import matplotlib.pyplot as plt


def  spectralnorm(A, rhoa=None):
    """
    Calculates the spectral norm of a Vector Auto-regressive coefficients
    matrix A. The spectral norm is the maximum singular value of A.

    Args:
        A (numpy array): Matrix of VAR coefficients
    
    Returns: 
        spectral_norm (float): Spectral norm of A
    """
    if np.ndim(A) == 1:
        p = len(A)
        p1 = p - 1
        A1 = np.vstack((A.T.reshape(1, -1), np.hstack((np.eye(p1), np.zeros((p1, 1))))))  # VAR coefficients for 1-lag problem
    elif np.ndim(A) == 2:
        shape = A.shape
        desired_num_dims = 3  # Number of dimensions you want to have
        # Extend the shape tuple with additional 1s if needed
        shape = shape + (1,) * (desired_num_dims - len(shape))
        n, n1, p = shape
        assert n1 == n, "VAR/VMA coefficients matrix has bad shape"
        pn1 = (p - 1) * n
        A1 = np.vstack((A.reshape(n, p * n), np.hstack((np.eye(pn1), np.zeros((pn1, n))))))  # VAR coefficients for 1-lag problem
        #A1.transpose((2,0,1))
    else:
        n, n1, p = A.shape
        assert n1 == n, "VAR/VMA coefficients matrix has bad shape"
        pn1 = (p - 1) * n
        A1 = np.vstack((A.reshape(n, p * n), np.hstack((np.eye(pn1), np.zeros((pn1, n))))))  # VAR coefficients for 1-lag problem

    eigenvalues = np.linalg.eigvals(A1)
    rho_intemediary = np.max(np.abs(eigenvalues))

    if rhoa is None:
        return rho_intemediary  # <-- spectral radius(norm)
    else:
        A = VAR_spectral_radius_decay(A, rhoa/rho_intemediary) # <-- adjusted coefficients
        return A, rho_intemediary # <-- previous spectral radius(norm)


def spectrallimit(A, M, r1, r2):
    """
    Finds the value of 'r' that satisfies the minimum phase (VMA) and stability (VAR) conditions.
    The minimum phase condition is that the spectral norm of the VMA coefficients is less than 1.
    The stability condition is that the spectral norm of the VAR coefficients is less than 1.

    Args:
        A (ndarray): The VMA coefficients.
        M (ndarray): The VAR coefficients.
        r1 (float): The lower bound of the search interval for 'r'.
        r2 (float): The upper bound of the search interval for 'r'.

    Returns:
        float: The value of 'r' that satisfies the minimum phase and stability conditions.
    """
    
    assert spectralnorm(A - r1 * M) > 1 and spectralnorm(A - r2 * M) < 1
    if isinstance(r1, (int, float)) and isinstance(r2, (int, float)):
        r = (r1 + r2) / 2
    else:
        raise TypeError("r1 and r2 must be numbers")

    while True:    
        rho = spectralnorm(A - r * M)
        
        if rho > 1:
            r1 = r
        else:
            r2 = r
        if np.abs(r1 - r2) < 1e-6:
            break
    return r


def VAR_spectral_radius_decay(A, decay_factor):
    if np.ndim(A) == 1:
        p = len(A)
        f = decay_factor
        for i in range(p):
            A[i] = f*A[i]
            f = decay_factor * f
        return A
    else:
        if np.ndim(A) == 2:
            A = A[:, :, np.newaxis]
            p = A.shape[2]
            f = decay_factor
            for i in range(p):
                A[:, :, i] = f * A[:, :, i]
                f = decay_factor * f
        else:
            p = A.shape[2]
            f = decay_factor
            for i in range(p):
                A[:, :, i] = f * A[:, :, i]
                f = decay_factor * f
        return A

def iss_rand(n, m, rhoa, dis=False):
    assert rhoa < 1

    if dis is None:
        dis = False

    A = spectralnorm(np.random.randn(m, m), rhoa=rhoa)
    C = np.random.randn(n, m)
    K = np.random.randn(m, n)

    M = K @ C
    rmin = spectrallimit(A[0], M, -1, 0)
    rmax = spectrallimit(A[0], M, 1, 0)

    r = rmin + (rmax - rmin) * np.random.rand()
    sqrtr = np.sqrt(np.abs(r))
    C = sqrtr * C
    K = np.sign(r) * sqrtr * K

    rhob = spectralnorm(A[0] - K @ C)

    return A, C, K, rhob

    if dis:
        nr = 1000
        ramax = 1.1 * max(rmax, -rmin)
        rr = np.linspace(-ramax, ramax, nr)
        rrhob = np.zeros(nr)
        for i in range(nr):
            rrhob[i] = spectralnorm(A - rr[i] * M)
        rholim = [0.9 * min(rrhob), 1.1 * max(rrhob)]
        plt.plot(rr, rrhob)
        plt.xlim([-ramax, ramax])
        plt.ylim(rholim)
        plt.axhline(y=1, color='k')
        plt.axvline(x=0, color='r')
        plt.axvline(x=r, color='g')
        plt.xlabel('r')
        plt.ylabel('rho')
        plt.legend(['rho(B)'])
        plt.title('rho(A) = %g, rho(B) = %g' % (rhoa, spectralnorm(A - r * M)))
        plt.show()
    return A, C, K, rhob

    

def symmetrise(x, hermitian=True, ut2lt=True):
    """
    Symmetrise a square matrix by reflecting the upper (default) or lower triangle
    across the diagonal. The default is to make a Hermitian matrix, which requires
    that the diagonal is real (if the entire matrix is real then the 'hermitian'
    flag has no effect). The diagonal is always left unchanged.
    
    Args:
    - x: np.ndarray, the input square matrix
    - hermitian: bool, optional, default is True. 
      Whether to return a Hermitian matrix (True) or a symmetric matrix (False)
    - ut2lt: bool, optional, default is True.
      Whether to symmetrise the upper triangle (True) or the lower triangle (False)
      
    Returns:
    - y: np.ndarray, the symmetrised matrix
    """
    n, n1 = x.shape
    assert np.isclose(n, n1) and n > 0, 'Input must be a square matrix'

    if ut2lt:
        tidx = np.tril_indices(n, -1) # indices of lower triangle
    else:
        tidx = np.triu_indices(n, +1) # indices of upper triangle

    y = x.copy()
    if hermitian:
        assert np.all(np.isreal(np.diag(x))), 'Diagonal must be real for a Hermitian matrix'
        x = x.conj().T
    else:
        x = x.T
    y[tidx] = x[tidx]
    return y


# NEED TO ADD DARE EQUATION SOLVER FROM MATLAB

def mdare(A, C, Q, R=None, S=None):
    """
    Compute innovations form parameters for a state space model in general form by
    solution of a discrete algebraic Riccati equation (DARE). This is a "stripped
    down" version of Matlab's dare function (real-valued only, no balancing).

    A, C, Q, R, S - general form state space parameters

    K             - Kalman gain matrix
    V             - innovations covariance matrix
    rep           - DARE report (see below)
    L             - DARE stablising eigenvalues
    P             - DARE solution

    The value returned in rep is negative if an unrecoverable error was detected:
    rep = -1 means that the DARE was found to have eigenvalues on (or near) the
    unit circle, while rep = -2 indicates that no stabilising solution to the DARE
    could be found. If no error occurred, rep returns the relative residual of the
    DARE solution, which should be tested for accuracy (rep > sqrt(eps) is
    reasonable).

    Note that if the SS (A,C,Q,R,S) is well-formed - that is, A is stable and R
    positive definite - then (A,K,V) should be a well-formed innovations-form SS
    model. WE DON'T TEST FOR THAT HERE! It is up to the caller to do so if deemed
    necessary.
    """
    r, r1 = A.shape
    assert r1 == r, "A must be a square matrix."
    n, r1 = C.shape
    assert r1 == r, "C must have the same number of rows as A."
    r1, r2 = Q.shape
    assert r1 == r and r2 == r, "Q must be a square matrix with the same dimensions as A."
    if R is None:
        R = np.eye(n)
    else:
        n1, n2 = R.shape
        assert n1 == n and n2 == n, "R must be a square matrix with the same dimensions as C."
    if S is None:
        S = np.zeros((r, n))
    else:
        r1, n1 = S.shape
        assert r1 == r and n1 == n, "S must have the same number of rows as A and the same number of columns as C."
    rr = 2 * r
    i = np.arange(rr)
    j = np.arange(r)
    k = np.arange(r, rr)
    K = np.empty((0, 0))
    V = np.empty((0, 0))
    P = np.empty((0, 0))

    # Solve the DARE using Generalized Schur (QZ) decomposition on the extended pencil:
    H = np.block([[A.T, np.zeros((r, r)), C.T], [-Q, np.eye(r), -S], [S.T, np.zeros((n, r)), R]])
    J = np.block([[np.eye(r), np.zeros((r, r + n))], [np.zeros((r, r)), A, np.zeros((r, n))],
                  [np.zeros((n, r)), -C, np.zeros((n, n))]])
    
        # QR decomposition - note: assumes real-valued, no balancing!

    q, _ = qr(H[:, rr:rr + n], mode='economic')

    # QZ algorithm

    H = q[:, i + n].T @ H[:, i]
    J = q[:, i + n].T @ J[:, i]
    JJ, HH, q, z = ordqz(J[i, i], H[i, i], output='real')
    JJ, HH, _, z[i, :] = ordqz(JJ, HH, q, z, output='complex', sort='udo')
    L = np.sort(np.linalg.eigvals(JJ, HH))[::-1]

    # Check for stable invariant subspace

    sis = np.abs(L) > 1
    if np.any(~sis[j, :]) or np.any(sis[k, :]):
        rep = -1
        return K, V, rep, L, P  # IMPORTANT: caller must test!!! (error is: ''DARE: eigenvalues on/near unit circle'')

    P1 = z[j, j]
    P2 = z[k, j]

    # Solve P = P2/P1

    PP, LL, UU = lu(P1)
    n = PP.shape[0]
    pvec = np.zeros(n, dtype=int)
    for i in range(n):
        pvec[i] = np.where(PP[:, i] == 1)[0][0]

    if np.linalg.cond(UU) < np.finfo(UU.dtype).eps:
        rep = -2
        return None, None, None, rep # IMPORTANT: caller must test!!! (error is: 'DARE: couldn''t find stabilising solution')
    P = np.linalg.solve(UU, np.linalg.solve(LL, P2.dot(P[:,pvec])))
    P = (P + P.T)/2
    
    U = A@P@C.T + S
    V = C@P@C.T + R
    K = U @ np.linalg.inv(V)
    
    APA = A @ P @ A.T - P
    UK = U@K.T
    rep = np.linalg.norm(APA - UK + Q, ord=1) / (1 + np.linalg.norm(APA, ord=1) + np.linalg.norm(UK, ord=1) + np.linalg.norm(Q, ord=1)) # relative residual

    # IMPORTANT: test for accuracy  - something like
    #
    # if rep > np.sqrt(eps):
    #     warning('DARE: possible inaccuracy (relative residual = %e)',rep);
    #

    
    L = L[k] # Return stable eigenvalues
    return K, V, rep, L, P




# FUNCTION DEFINES THE SS ERROR MESSAGES
def sserror(rep, y=None, tol=np.sqrt(np.finfo(float).eps)):
    err = rep < 0  # DARE failed: show-stopper!

    if y is None:
        if err:
            sys.stderr.write('DARE ERROR: ')
            if rep == -1:
                sys.stderr.write('eigenvalues on/near unit circle\n')
            elif rep == -2:
                sys.stderr.write('no stabilizing solution\n')
            return

        if rep > tol:
            sys.stderr.write('DARE WARNING: large relative residual = %e\n' % rep)
    else:
        if err:
            sys.stderr.write('DARE ERROR for source %d: ' % y)
            if rep == -1:
                sys.stderr.write('eigenvalues on/near unit circle\n')
            elif rep == -2:
                sys.stderr.write('no stabilizing solution\n')
            return

        if rep > tol:
            sys.stderr.write('DARE WARNING for source %d: large relative residual = %e\n' % (y, rep))




def ss_parms(A, C, K, V=None):
    r, r1 = A.shape
    assert r1 == r, "SS: bad 'A' parameter"
    n, r1 = C.shape
    assert r1 == r, "SS: bad 'C' parameter"
    r1, n1 = K.shape
    assert n1 == n and r1 == r, "SS: bad 'K' parameter"
    
    L = None
    
    if V is not None:
        n1, n2 = V.shape
        assert n1 == n and n2 == n, "SS: bad 'V' parameter"

    L = np.linalg.cholesky(V).T
    if np.any(np.isnan(L)):
        L = None  # not positive-definite; caller to test
    
    return n, r, L



# FUNCTION: STATE-SPACE MODEL TO PAIRWISE GRANGER CAUSALITY
def ss_to_pwcgc(A, C, K, V):
    n, _, L = ss_parms(A, C, K, V)
    F = np.full((n, n), np.nan)
    KL = K @ L
    KVK = KL @ KL.T
    LDV = np.log(np.diag(V))
    
    for y in range(n):
        r = np.concatenate((np.arange(y), np.arange(y + 1, n)))  # omit y
        
        _, VR, rep = mdare(A, C[r, :], KVK, V[r, r], K @ V[:, r])  # "reduced" innovations covariance
        if sserror(rep, y):
            continue  # check DARE report, bail out on error
        
        F[r, y] = np.diag(np.log(np.diag(VR))) - LDV[r]
    
    return F

def transform_ss(A, C, K, V):
    n = C.shape[0]
    SQRTV = cholesky(V, lower=True)
    ISQRTV = inv(SQRTV)
    C = ISQRTV @ C
    K = K @ SQRTV
    V = np.eye(n)
    return A, C, K, V

def iss2cak(A, C, K):
    r = A.shape[0]
    n = C.shape[0]
    Ak = np.eye(r)
    CAK = np.zeros((n, n, r))
    CAK[:, :, 0] = C @ K
    for k in range(1, r):
        Ak = Ak @ A
        CAK[:, :, k] = C @ Ak @ K
    return CAK