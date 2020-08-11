import numpy as np
import scipy.linalg as la
import aa1s
import aa

### l1 regularized least squares with ADMM  ###
# data and hyper-parameters
np.random.seed(1)
n = 300
m = 100
mu = 0.1
rho = 0.1
N = 5000
A = np.random.randn(m,n)
b = np.random.randn(m)
L = la.cho_factor(A.T.dot(A) + rho * np.identity(n))

def soft_thresh(y, t):
    return np.sign(y) * np.maximum(abs(y) - t, 0)

z0 = np.linalg.solve(A.T.dot(A), A.T.dot(b))

z = z0
lam = np.zeros(n)

# XXX: WIP

def l1ls_admm(lam):
    x = la.cho_solve(L, rho * (z + lam) + A.T.dot(b))
    z = soft_thresh(x - lam, mu / rho)
    lam = lam - x + z

# aa-i-s
mem = 10
type1 = 1
params = {'theta':0, 'tau':0, 'D':1e6, 'eps':1e-6, 'beta_0':0}
aa1s_wrk = aa1s.AndersonAccelerator(n, mem, params, type1)

print('### aa-i-s, least squares gradient descent ###')
#x = np.zeros((n,1))
x = np.zeros(n)
for i in range(300):
    x = aa1s_wrk.apply(ls_gd, x)
    #x = ls_gd(x)
    if i % 10 == 0:
        print('i: ', i,' err: ', np.linalg.norm(A.T.dot(A.dot(x) - b)))
print()

# original aa1
print('### vanilla aa1, least squares gradient descent ###')
aa_wrk = aa.AndersonAccelerator(n, mem, type1)
x = np.zeros(n)
for i in range(N):
    x_prev = np.copy(x)
    x = ls_gd(x)
    aa_wrk.apply(x, x_prev)
    if i % 10 == 0:
        print('i: ', i,' err: ', np.linalg.norm(A.T.dot(A.dot(x) - b)))

