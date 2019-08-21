from __future__ import print_function
import numpy as np
import aa as aa

dim = 1000
mem = 20
type1 = 1

np.random.seed(1)

x = np.random.randn(dim)
Q = np.random.randn(dim,dim)
Q = Q.T.dot(Q)

step = 0.0005

aa_wrk = aa.AndersonAccelerator(dim, mem, type1)

for i in xrange(1000):
    x_prev = np.copy(x)
    x -= step * Q.dot(x)
    aa_wrk.apply(x, x_prev)
    if i % 100 == 0:
        print('i: ', i,' err: ', np.linalg.norm(x))
