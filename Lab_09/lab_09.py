import matplotlib.pyplot as plt
import numpy as np
import math



def setVectorC(N):
    N = int(N / 2) + 1
    c = np.zeros(N)
    for i in range(N):
        c[i] = (-1) ** i / math.factorial(i)
    return c


def setMatrixA(M, N, c):
    M = int(M / 2)
    N = int(N / 2)
    A = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            A[i][j] = c[N - M + i + j + 1]
    return A


def setVectorY(N, M, c):
    N = int(N / 2)
    M = int(M / 2)
    y = np.zeros(M)
    for i in range(M):
        y[i] = -c[N + 1 + i]
    return y


def setVectorB(x, M):
    M = int(M / 2)
    b = np.zeros(M + 1)
    b[0] = 1
    for i in range(M):
        b[M - i] = x[i]
    return b


def setVectorA(b, c, N):
    a = np.zeros(int(N / 2) + 1)
    for i in range(int(N / 2) + 1):
        for j in range(i + 1):
            a[i] += c[i - j] * b[j]
    return a


def setPolyR(P, Q, x):
    sumOne = sumTwo = 0.
    for i in range(len(P)):
        sumOne += P[i] * math.pow(x, 2 * i)
    for i in range(len(Q)):
        sumTwo += Q[i] * math.pow(x, 2 * i)
    return sumOne / sumTwo


def exe(M, N):
    # data
    R = []
    delta_x = 0.01
    begin = -5
    end = 5

    c = setVectorC(N + M)
    A = setMatrixA(M, N, c)
    y = setVectorY(N, M, c)
    x = np.linalg.solve(A, y)
    b = setVectorB(x, M)
    a = setVectorA(b, c, N)

    x = np.arange(begin, end, delta_x)
    for i in x:
        R.append(setPolyR(a, b, i))

    # print('b:', b)
    # print('a:', a)
    # print('c:', c)
    # print('A:', A)
    # print('x:', x)
    # print('y:', y)
    # print('R:', R)
    # -------------------------------- CHART --------------------------------
    plt.figure()
    plt.plot(x, R, label=r'$ R_{ %(M)d, %(N)d } (x)$' % {'M': N, 'N': M}, color='tab:green')
    plt.plot(x, np.exp(-x ** 2), label=r'$exp(-x^2)$', color='tab:pink')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    arrNM = [[2, 2], [4, 4], [6, 6], [2,4], [2,6], [2,8]]
    for j, i in arrNM:
        print(f"Wartosci dla M = {i}, N ={j}")
        exe(i, j)
