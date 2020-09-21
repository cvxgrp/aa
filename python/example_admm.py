import numpy as np
import scipy.linalg as la
import aa1s
import aa
import matplotlib.pyplot as plt

### l1 regularized least squares with ADMM  ###
# data and hyper-parameters
dim = 300
mem = 100
type1 = 1
iter_max = 5000
print_gap = 50

np.random.seed(1)

m = 100
mu = 0.1
rho = 0.1
A = np.random.randn(m,dim)
b = np.random.randn(m)
L = la.cho_factor(A.T.dot(A) + rho * np.identity(dim))

z0 = np.linalg.solve(A.T.dot(A), A.T.dot(b))

def soft_thresh(y, t):
    return np.sign(y) * np.maximum(abs(y) - t, 0)

def l1ls_admm_res(u):
    z, lam = u[:dim], u[dim:]
    z_old = np.copy(z)
    x = la.cho_solve(L, rho * (z + lam) + A.T.dot(b))
    z = soft_thresh(x - lam, mu / rho)
    lam = lam - x + z
    return np.linalg.norm(x-z), np.linalg.norm(z-z_old)


def l1ls_admm(u):
    z, lam = u[:dim], u[dim:]
    x = la.cho_solve(L, rho * (z + lam) + A.T.dot(b))
    z = soft_thresh(x - lam, mu / rho)
    lam = lam - x + z
    u[:dim] = z
    u[dim:] = lam
    return u

params = {'theta':0.01, 'tau':0.001, 'D':1e6, 'eps':1e-6, 'beta_0':0.001, 'fp': l1ls_admm}

# gd
print('### admm: l1-regularized least-squares with ADMM ###')
z = np.copy(z0)
lam = np.zeros(dim)
u = np.hstack([z, lam])
res_admm = []
ds_admm = []
for i in range(iter_max):
    # evaluate residuals
    ps, ds = l1ls_admm_res(u)
    res_admm.append(ps)
    ds_admm.append(ds)
    # update the iterate
    u = l1ls_admm(u)
    z, lam = u[:dim], u[dim:]
    if i % print_gap == 0:
        print('i: ', i+1,' primal res: ', res_admm[-1], 'dual res: ', ds_admm[-1])
if i % print_gap != 0:
    print('i: ', i+1,' primal res: ', res_admm[-1], 'dual res: ', ds_admm[-1])
print()

# aa-i-s
print('### aa-i-s: l1-regularized least-squares with ADMM ###')
aa1s_wrk = aa1s.AndersonAccelerator(2*dim, mem, params, type1)
z = np.copy(z0)
lam = np.zeros(dim)
u = np.hstack([z, lam])
res_aa1s = []
ds_aa1s = []
for i in range(iter_max):
    # evaluate residuals
    ps, ds = l1ls_admm_res(u)
    res_aa1s.append(ps)
    ds_aa1s.append(ds)
    # update the iterate
    u_old = np.copy(u)
    u = l1ls_admm(u)
    aa1s_wrk.apply(u, u_old)
    z, lam = u[:dim], u[dim:]
    if i % print_gap == 0:
        print('i: ', i+1,' primal res: ', res_aa1s[-1], 'dual res: ', ds_aa1s[-1])
if i % print_gap != 0:
    print('i: ', i+1,' primal res: ', res_aa1s[-1], 'dual res: ', ds_aa1s[-1])
print()

# original aa1
print('### vanilla aa1: l1-regularized least-squares with ADMM ###')
#type1 = 0 # switch to type2
aa_wrk = aa.AndersonAccelerator(2*dim, mem, type1)
z = np.copy(z0)
lam = np.zeros(dim)
u = np.hstack([z, lam])
res_aa1 = []
ds_aa1 = []
for i in range(iter_max):
    # evaluate residuals
    ps, ds = l1ls_admm_res(u)
    res_aa1.append(ps)
    ds_aa1.append(ds)
    # update the iterate
    u_old = np.copy(u)
    u = l1ls_admm(u)
    aa_wrk.apply(u, u_old)
    z, lam = u[:dim], u[dim:]
    if i % print_gap == 0:
        print('i: ', i+1,' primal res: ', res_aa1[-1], 'dual res: ', ds_aa1[-1])
if i % print_gap != 0:
    print('i: ', i+1,' primal res: ', res_aa1[-1], 'dual res: ', ds_aa1[-1])
print()

plt.semilogy(res_admm, label='admm - res')
plt.semilogy(ds_admm, label='admm - dual')
plt.semilogy(res_aa1s, label='aa1s - res')
plt.semilogy(ds_aa1s, label='aa1s -dual')
plt.semilogy(res_aa1, label='aa1 - res')
plt.semilogy(ds_aa1, label='aa1 -dual')
plt.legend()
plt.show()

