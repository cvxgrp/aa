# min (1/2) x'Q'x - q'x

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import aa

dim = 100
mems = [5, 10, 20, 50]
N = int(1e4)

np.random.seed(1234)

Q = np.random.randn(dim, dim)
Q = 0.1 * Q.T.dot(Q)
q = np.random.randn(dim)
x_0 = np.random.randn(dim)
x_star = np.linalg.solve(Q, q)

step = 1.0 / np.max(np.linalg.eigvals(Q))

f = lambda x: 0.5 * x.T @ Q @ x - q.T @ x

f_star = f(x_star)
print('f^* = ', f_star)

print('No acceleration')

results = {}

fs = []
x = x_0.copy()
for i in range(N):
    x_prev = np.copy(x)
    x -= step * (Q.dot(x) - q)
    fs.append(f(x) - f_star)
    if i % 1000 == 0:
        print('i: ', i,' f - f^*: ', np.abs(f(x) - f_star))

results['No accel'] = fs

RELAXATION = 1.0

for mem in mems:
  print('Type-I acceleration, mem:', mem)
  fs = []
  x = x_0.copy()
  aa_wrk = aa.AndersonAccelerator(dim, mem, True, regularization=1e-8,
                                  relaxation=RELAXATION, verbosity=1)
  for i in range(N):
      if i > 0: aa_wrk.apply(x, x_prev)
      x_prev = np.copy(x)
      x -= step * (Q.dot(x) - q)
      aa_wrk.safeguard(x, x_prev)
      fs.append(f(x) - f_star)
      if i % 1000 == 0:
          print('i: ', i,' f - f^*: ', np.abs(f(x) - f_star))

  results[f'AA-I {mem}'] = fs

  print('Type-II acceleration, mem:', mem)
  fs = []
  x = x_0.copy()
  aa_wrk = aa.AndersonAccelerator(dim, mem, False, regularization=1e-10,
                                  relaxation=RELAXATION, verbosity=1)
  for i in range(N):
      if i > 0: aa_wrk.apply(x, x_prev)
      x_prev = np.copy(x)
      x -= step * (Q.dot(x) - q)
      aa_wrk.safeguard(x, x_prev)
      fs.append(f(x) - f_star)
      if i % 1000 == 0:
          print('i: ', i,' f - f^*: ', np.abs(f(x) - f_star))

  results[f'AA-II {mem}'] = fs

for k,v in results.items():
    plt.semilogy(v, label=k)

plt.legend()
plt.show()
