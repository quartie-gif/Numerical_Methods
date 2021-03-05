import numpy as np
import math
import matplotlib.pyplot as plt


def gauss(_A, _b):
    n = _A.shape[0]
    for i in range(0,n-1):
        for k in range(i+1,n):
            l=(_A[k,i]/_A[i,i])
            _A[k] = _A[k]-_A[i]*l
            _b[k] = _b[k]-_b[i]*l
    _x = [0]*n
    _x[n-1] = _b.item(n-1)/_A.item(n-1,n-1)
    for k in range(n-2,-1,-1):
        _x[k] = (_b[k]-sum(_A[k]*np.matrix(_x).T)).item(0)/_A.item(k,k)
    return np.matrix(_x).T

labelx = []
labely = []
for q in np.arange(1 / 5, 5, 0.000000000001):
    A = np.array([[q * 2 * pow(10, -4), 1, 6, 9, 10], [2 * pow(10, -4), 1, 6, 9, 10], [1, 6, 6, 8, 6], [5, 9, 10, 7, 10], [3, 4, 9, 7, 9]],
                 float)
    b = np.array([10, 2, 9, 9, 3], float)

    solution = gauss(A,b)

    c = A*solution
    suma = 0
    for i in range(5):
        suma += (c[i]-b[i])**2
    wynik = 1/5*math.sqrt(suma)

    labelx.append(q)
    labely.append(wynik)

    print(wynik)

print(labelx,labely)
plt.plot(labelx,labely)
plt.ylabel('wychylenie w skali logarytmicznej ')
plt.yscale('log')
plt.xlabel('q')
plt.show()