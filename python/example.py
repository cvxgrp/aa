# min (1/2) x'Q'x - q'x

from __future__ import print_function
import numpy as np
import aa

dim = 1000
mems = [5, 10, 20, 50, 100]
N = int(1e4)

np.random.seed(1234)

Q = np.random.randn(dim,dim)
Q = Q.T.dot(Q)
q = np.random.randn(dim)
x_0 = np.random.randn(dim)
x_star = np.linalg.solve(Q, q)

step = 0.0005

def f(x):
  return 0.5 * x.T @ Q @ x - q.T @ x

f_star = f(x_star)
print('f^* = ', f_star)

print('No acceleration')
x = x_0.copy()
for i in range(N):
    x_prev = np.copy(x)
    x -= step * (Q.dot(x) - q)
    if i % 1000 == 0:
        print('i: ', i,' f - f^*: ', f(x) - f_star)

for mem in mems:
  print('Type-I acceleration, mem:', mem)
  x = x_0.copy()
  aa_wrk = aa.AndersonAccelerator(dim, mem, True, eta=1e-8)
  for i in range(N):
      x_prev = np.copy(x)
      x -= step * (Q.dot(x) - q)
      aa_wrk.apply(x, x_prev)
      if i % 1000 == 0:
          print('i: ', i,' f - f^*: ', f(x) - f_star)

  print('Type-II acceleration, mem:', mem)
  x = x_0.copy()
  aa_wrk = aa.AndersonAccelerator(dim, mem, False, eta=1e-10)
  for i in range(N):
      x_prev = np.copy(x)
      x -= step * (Q.dot(x) - q)
      aa_wrk.apply(x, x_prev)
      if i % 1000 == 0:
          print('i: ', i,' f - f^*: ', f(x) - f_star)
