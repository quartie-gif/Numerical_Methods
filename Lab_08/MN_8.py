import matplotlib.pyplot as plt
import numpy as np


def setX(size, begin, end):
    step = (end - begin) / (size - 1)
    x = [begin + (step * i) for i in range(size)]
    return x


def getX(x, index):
    return x[index]


def setD(n, x, alpha, beta, y):
    d = np.zeros(n)
    d[0] = alpha
    d[-1] = beta
    for i in range(1, n - 1):
        d[i] = 6 / (getH(x, i) + getH(x, i + 1)) * (
                (y[i + 1] - y[i]) / getH(x, i + 1) - (y[i] - y[i - 1]) / getH(x, i))
    return d

def getH(arr, index):
    return getX(arr, index) - getX(arr, index - 1)


def lamb(arr, index):
    return 1 - getH(arr, index + 1) / (getH(arr, index) + getH(arr, index + 1))


def mi(arr, index):
    return 1 - lamb(arr, index)


def setM(A, d):
    return np.linalg.solve(A, d)


def setA(n, x):
    A = np.zeros(shape=(n, n))

    for i in range(1, n - 1):
        A[i][i - 1] = mi(x, i)
        A[i][i + 1] = lamb(x, i)
        A[i][i] = 2.0

    A[0][0] = 1.0
    A[0][1] = 0.0
    A[n - 1][n - 1] = 1.0
    A[n - 1][n - 2] = 0.0

    return A


def getA(y, m, i):
    return (y[i] - y[i - 1]) / getH(x, i) - getH(x, i) / 6 * (m[i] - m[i - 1])


def getB(x, y, i, m):
    return y[i - 1] - m[i - 1] * getH(x, i) ** 2 / 6


def getS(y, x, m, X, n):
    i = 0
    for j in range(n - 1):
        if x[j] <= X <= x[j + 1]:
            i = j + 1
    return m[i - 1] * pow((x[i] - X), 3) / (6 * getH(x, i)) + m[i] * pow((X - x[i - 1]), 3) / (
            6 * getH(x, i)) + getA(y, m, i) * (X - x[i - 1]) + getB(x, y, i, m)


def fun1(x, n):
    y = [1 / (1 + x[i] ** 2) for i in range(n)]
    return y


def fun2(x, n):
    y = [np.cos(2 * x[i]) for i in range(n)]
    return y


def getSecondDerivative(fun):
    dx = 0.01
    return [(fun(x - dx) - 2 * fun(x) + fun(x + dx)) / pow(dx, 2) for x in np.arange(begin, end, 0.01)]


if __name__ == "__main__":
    # Data
    n = 21  # ilosc wezlow

    begin = -5
    end = 5

    x = setX(n, begin, end)
    y = fun1(x, n)
    A = setA(n, x)
    d = setD(n, x, 0, 0, y)
    m = setM(A, d)
    s = [getS(y, x, m, i, n) for i in x]
    print(m)
    print(x)
    print(d)
    print(A)
    print(s)

    foo1 = lambda x: 1 / (1 + x ** 2)
    foo2 = lambda x: np.cos(2 * x)

    derivative = getSecondDerivative(foo1)

    plt.plot(np.arange(begin, end, 0.01), [getS(y, x, m, i, n) for i in np.arange(begin, end, 0.01)], linestyle='-',
             label='funkcja interpolująca')
    plt.plot([i for i in x], [i for i in s], linestyle='', marker="^",)
    # plt.plot([i for i in x], [i for i in m],color='darkgreen', linestyle='', marker="^", label='wektor m')
    # plt.plot(np.arange(begin, end, 0.01), [i for i in derivative], linestyle='-', label='pochodne dokladne')
    plt.plot(np.arange(begin, end, 0.01), [foo1(x) for x in np.arange(begin, end, 0.01)], label='y = f(x)')
    # plt.plot(np.arange(begin, end, 0.01), [foo2(x) for x in np.arange(begin, end, 0.01)], label='y = f(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend( loc='upper left')
    plt.show()
