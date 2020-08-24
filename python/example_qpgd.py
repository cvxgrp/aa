import numpy as np
import aa1s
import aa

### quadratic gradient descent ###
# data and hyper-parameters
n = 2#10
np.random.seed(1)
x0 = np.random.randn(n)
Q = np.random.randn(n,n)
Q = Q.T.dot(Q)
step = 0.0005 # step = 0.05, gd diverge, but aa1s converge with mem = 20

def gd(x):
	return x - step * Q.dot(x)

dim = n
mem = 5
type1 = 1
params = {'theta':0, 'tau':0, 'D':1e6, 'eps':1e-6, 'beta_0':0}

# # aa-i-s
# aa1s_wrk = aa1s.AndersonAccelerator(dim, mem, params, type1)
# x = np.copy(x0)
# for i in range(10):
#     x = aa1s_wrk.apply(gd, x)
#     #x = gd(x)
#     if i % 1 == 0:
#         print('i: ', i,' err: ', np.linalg.norm(x))

# debug: one step of vanilla aa1
x = np.copy(x0)
x1 = gd(x)
g0 = x - x1
s0 = x1 - x
x2 = gd(x1)
g1 = x1 - x2
y0 = g1 - g0
x2_aa = x1 - g1 - (s0-y0)*np.dot(s0,g1)/np.dot(s0,y0)
print('Sg = {}'.format(np.dot(s0,g1)))
print('SY divide Sg = {}'.format(np.dot(s0,g1)/np.dot(s0,y0)))
print('D = {}'.format(s0-y0))
print('s0 = {}; \ny0 = {}; \ng1 = {}'.format(s0, y0, g1))
print()
print('### aa-i-s results ###')
#x2_aa = x1 - g1 - (x2-x1)*np.dot(s0,g1)/np.dot(s0,y0)
#print('err of x: ', np.linalg.norm(x))
print('i:  0, err: ', np.linalg.norm(x1))
# print('err of x2: ', np.linalg.norm(x2))
print('i:  1, err: ', np.linalg.norm(x2_aa))
print()


# original aa1 implementation
print('### vanilla aa1 results ###')
aa_wrk = aa.AndersonAccelerator(dim, mem, type1)
x = np.copy(x0)
for i in range(2):
	x_prev = np.copy(x)
	#x -= step * Q.dot(x)
	x = gd(x)
	print('x = {}; x_prev = {}.'.format(x, x_prev))
	aa_wrk.apply(x, x_prev)
	print('i: ', i, ' err: ', np.linalg.norm(x))

