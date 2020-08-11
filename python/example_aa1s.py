import numpy as np
import aa1s

dim = 1000
mem = 20
type1 = 1

np.random.seed(1)

x = np.random.randn(dim)
Q = np.random.randn(dim,dim)
Q = Q.T.dot(Q)

step = 0.0005 # step = 0.05, gd diverge, but aa1s converge with mem = 20

params = {'theta':0.01, 'tau':0.001, 'D':1e6, 'eps':1e-6}

aa1s_wrk = aa1s.AndersonAccelerator(dim, mem, params, type1)

def gd(x):
	return x - step * Q.dot(x)

for i in range(10000):
    x = aa1s_wrk.apply(gd, x)
    #x = gd(x)
    if i % 100 == 0:
        print('i: ', i,' err: ', np.linalg.norm(x))
