import numpy as np
import aa1s
import aa
import matplotlib.pyplot as plt
import copy

### quadratic gradient descent ###
# data and hyper-parameters
dim = 1000
mem = 10
type1 = 1
iter_max = 10000
print_gap = 100

np.random.seed(1)

x0 = np.random.randn(dim)
Q = np.random.randn(dim,dim)
Q = Q.T.dot(Q)
step = 0.0005

def gd(x):
	return x - step * Q.dot(x)

params = {'theta':0, 'tau':0, 'D':1e6, 'eps':1e-6, 'beta_0':0, 'fp': gd}

x0_norm = np.linalg.norm(x0)

# gd
print('### gd: simple QP gradient descent')
x = np.copy(x0)
rec_gd = [x0_norm]
print('i: 0 err: ', np.linalg.norm(x))
for i in range(iter_max):
	x = gd(x)
	x_norm = np.linalg.norm(x)
	rec_gd.append(x_norm)
	if i % print_gap == 0:
		print('i: ', i+1, ' err: ', x_norm)
if i % print_gap != 0:
    print('i: ', i+1,' err: ', x_norm)
print()

# aa-i-s
print('### aa-i-s: simple QP gradient descent')
aa1s_wrk = aa1s.AndersonAccelerator(dim, mem, params, type1)
x = np.copy(x0)
rec_aa1s = [x0_norm]
print('i: 0 err: ', np.linalg.norm(x))
for i in range(iter_max):
	x_prev = np.copy(x)
	x = gd(x)
	aa1s_wrk.apply(x, x_prev)
	x_norm = np.linalg.norm(x)
	rec_aa1s.append(x_norm)
	if i % print_gap == 0:
		print('i: ', i+1, ' err: ', x_norm)
if i % print_gap != 0:
    print('i: ', i+1,' err: ', x_norm)
print()

# original aa1 implementation
print('### vanilla aa1: simple QP gradient descent')
aa_wrk = aa.AndersonAccelerator(dim, mem, type1)
x = np.copy(x0)
rec_aa1 = [x0_norm]
print('i: 0 err: ', np.linalg.norm(x))
for i in range(iter_max):
	x_prev = np.copy(x)
	x = gd(x)
	aa_wrk.apply(x, x_prev)
	x_norm = np.linalg.norm(x)
	rec_aa1.append(x_norm)
	if i % print_gap == 0:
		print('i: ', i+1, ' err: ', x_norm)
if i % print_gap != 0:
    print('i: ', i+1,' err: ', x_norm)

# plot the results
plt.semilogy(rec_gd)
plt.semilogy(rec_aa1s)
plt.semilogy(rec_aa1)
plt.legend(['gd', 'aa1s', 'aa1'])
plt.show()

