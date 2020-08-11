import numpy as np
import aa1s

dim = 10
mem = 5#20
type1 = 1

np.random.seed(1)

x = np.random.randn(dim)
Q = np.random.randn(dim,dim)
Q = Q.T.dot(Q)

step = 0.0005 # step = 0.05, gd diverge, but aa1s converge with mem = 20

params = {'theta':0, 'tau':0, 'D':1e6, 'eps':1e-6, 'beta_0':0}

aa1s_wrk = aa1s.AndersonAccelerator(dim, mem, params, type1)

def gd(x):
	return x - step * Q.dot(x)

for i in range(10):
    x = aa1s_wrk.apply(gd, x)
    #x = gd(x)
    if i % 1 == 0:
        print('i: ', i,' err: ', np.linalg.norm(x))
