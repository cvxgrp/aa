import numpy as np
import aa1s
import aa
import matplotlib.pyplot as plt
import copy

### https://docs.python.org/3/library/warnings.html
#import warnings
#warnings.simplefilter('ignore')

### least-squares gradient descent ###
# data and hyper-parameters
dim = 100 #500
mem = 10
type1 = 1
iter_max = 300
print_gap = 50

np.random.seed(1)

m = 300 # 750
A = np.random.randn(m,dim)
#b = np.random.randn(m,1)
b = np.random.randn(m)
t = 0.0002 
## when m = 750, n = 500
# t = 0.002: gd diverge; 
# t = 0.0002 and t = 0.00002, gd and vanilla aa1 almost the same results
## when m = 300, n = 100;
# t = 0.002: almost the same aa1s and vanilla a1
# t = 0.0002: aa1s much better, but after getting to 1e-13 residuals, NaNs appear -- try to solve it

def ls_gd(x):
    return x - t * A.T.dot(A.dot(x) - b)

params = {'theta':0, 'tau':0, 'D':1e6, 'eps':1e-6, 'beta_0':0, 'fp': ls_gd}

# gd
print('### gd: least squares gradient descent')
x = np.zeros(dim)
err = np.linalg.norm(A.T.dot(A.dot(x) - b))
rec_gd = [err]
for i in range(iter_max):
    x = ls_gd(x)
    err = np.linalg.norm(A.T.dot(A.dot(x) - b))
    rec_gd.append(err)
    if i % print_gap == 0:
        print('i: ', i+1,' err: ', err)
if i % print_gap != 0:
    print('i: ', i+1,' err: ', err)
print()


# aa-i-s
print('### aa-i-s, least squares gradient descent ###')
aa1s_wrk = aa1s.AndersonAccelerator(dim, mem, params, type1)
#x = np.zeros((n,1))
x = np.zeros(dim)
err = np.linalg.norm(A.T.dot(A.dot(x) - b))
rec_aa1s = [err]
print('i: 0 err:', err)
for i in range(iter_max):
    x_prev = np.copy(x)
    x = ls_gd(x)
    aa1s_wrk.apply(x, x_prev)
    err = np.linalg.norm(A.T.dot(A.dot(x) - b))
    rec_aa1s.append(err)
    if i % print_gap == 0:
        print('i: ', i+1,' err: ', err)
if i % print_gap != 0:
    print('i: ', i+1,' err: ', err)
print()

# original aa1
print('### vanilla aa1, least squares gradient descent ###')
aa_wrk = aa.AndersonAccelerator(dim, mem, type1)
x = np.zeros(dim)
err = np.linalg.norm(A.T.dot(A.dot(x) - b))
rec_aa1 = [err]
print('i: 0 err:', err)
for i in range(iter_max):
    x_prev = np.copy(x)
    x = ls_gd(x)
    aa_wrk.apply(x, x_prev)
    err = np.linalg.norm(A.T.dot(A.dot(x) - b))
    rec_aa1.append(err)
    if i % print_gap == 0:
        print('i: ', i,' err: ', err)
if i % print_gap != 0:
    print('i: ', i+1,' err: ', err)

# plot the results
plt.semilogy(rec_gd)
plt.semilogy(rec_aa1s)
plt.semilogy(rec_aa1, linestyle = '-.')
plt.legend(['gd', 'aa1s', 'aa1'])
plt.show()

