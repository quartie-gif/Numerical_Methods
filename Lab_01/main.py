import numpy as np
import math
import matplotlib.pyplot as plt


def gauss_solver(_A, _b):
    n = _A.shape[0]
    for i in range(0, n - 1):
        for k in range(i + 1, n):
            ratio = (_A[k, i] / _A[i, i])
            _A[k] -= _A[i] * ratio
            _b[k] -= _b[i] * ratio
    _x = np.zeros(n)
    _x[n - 1] = b[n - 1] / _A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        sum_j = 0
        for j in range(i + 1, n):
            sum_j += _A[i, j] * _x[j]
        _x[i] = (_b[i] - sum_j) / _A[i, i]
    return np.matrix(_x).T


labelx = []
labely = []

for q in np.arange(1 / 5, 5, 0.1000000234211):

    A = np.array([[q * 2e-4, 1, 6, 9, 10], [2e-4, 1, 6, 9, 10], [1, 6, 6, 8, 6], [5, 9, 10, 7, 10], [3, 4, 9, 7, 9]],
                 float)
    b = np.array([10, 2, 9, 9, 3], float)

    solution = gauss_solver(A, b)

    c = A @ solution

    Sum = 0

    for i in range(5):
        Sum += (c[i] - b[i]) ** 2
    variation = 1 / 5 * math.sqrt(Sum)

    labelx.append(q)
    labely.append(variation)
    print(variation)

print(labelx, labely)
plt.plot(labelx, labely)
plt.ylabel('wychylenie')
plt.yscale('log')
plt.xlabel('q')
plt.show()
