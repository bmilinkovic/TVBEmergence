import numpy as np
from scipy.linalg import subspace_angles
from mvgc import symmetrise
from itertools import combinations

def gmetric(L1, L2, maxangle=False):
    """Metric on the Grassmanian manifold, normalized to lie in [0,1]"""

    theta = subspace_angles(L1, L2)
    
    if maxangle:
        d = np.max(theta) / (np.pi / 2)
    else:
        d = np.sqrt(np.mean(theta**2)) / (np.pi / 2)  # default
        
    return d

def gmetrics(L):
    N = L.shape[2]
    d = np.zeros((N, N))
    for r1 in range(N):
        for r2 in range(r1+1, N):
            d[r1, r2] = gmetric(L[:, :, r1], L[:, :, r2])
    d = symmetrise(d)
    return d

def gmetricsx(L):
    n = L.shape[0]
    d = np.zeros(n)
    for i in range(n):
        v = np.zeros((n,1))
        v[i] = 1
        d[i] = gmetric(L, v)
    return d

def gmetricsxx(L):
    n, m = L.shape

    c = list(combinations(range(n), m))  # combinations
    nc = len(c)  # number of combinations
    d = np.zeros(nc)
    
    for k in range(nc):
        v = np.zeros((n, m))
        v[c[k], :] = np.eye(m)
        d[k] = gmetric(L, v, True)
        
    return d