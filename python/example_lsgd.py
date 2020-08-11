import numpy as np
import aa1s
import aa

### least-squares gradient descent ###
# data and hyper-parameters
np.random.seed(1)
n = 100
m = 300
A = np.random.randn(m,n)
#b = np.random.randn(m,1)
b = np.random.randn(m)
N = 300
t = 0.002

def ls_gd(x):
    return x - t * A.T.dot(A.dot(x) - b)

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

