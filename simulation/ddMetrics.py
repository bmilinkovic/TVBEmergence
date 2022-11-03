import itertools

import numpy
import scipy.linalg
import scipy
import math


def gmetric(L1, L2, maxangle=False):
    theta = scipy.linalg.subspace_angles(L1, L2)
    if maxangle:
        d = numpy.max(theta) / (numpy.pi / 2)
    else:
        d = numpy.sqrt(numpy.mean(theta**2)) / (numpy.pi / 2)
    return d

def gmetrics(L):
    N = numpy.size(L, axis=0)
    d = numpy.zeros((N,N))
    for r1 in range(N):
        for r2 in range(r1+1, N):
            d[r1,r2] = gmetric(L[r1,:,:], L[r2,:,:]) # this could be wrong, check how to index 3D arrays in python
    return d

def gmetricsx(L):
    n = numpy.size(L, axis=1)
    d = numpy.zeros((n,1))
    for i in range(n):
        v = numpy.zeros((n,1))
        v[i] = 1
        d[i] = gmetric(L,v)
    return d



# NOT WORKING: SOMETHING TO DO WITH INDEXING

# def gmetricsxx(L):
#     n, m = numpy.shape(L)
#     binom = list(itertools.combinations(numpy.arange(1,n+1,1),m))
#     nchoosek = [item for t in binom for item in t]
#     nchoosek = numpy.reshape(nchoosek, (21,5))
#     nc = numpy.size(nchoosek, axis=0)
#     d = numpy.zeros((nc, 1))
#     for k in range(nc - 1):
#         v = numpy.zeros((n, m))
#         v[nchoosek[k,:], :] = [numpy.eye(m, m)]
#         d[k] = gmetric(L, v, True)
#     return d, nchoosek







