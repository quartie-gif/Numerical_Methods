import matplotlib.pyplot as plt
import numpy as np


def setX(begin, end, size):
    step = (end - begin) / (size - 1)
    x = [begin + (step * i) for i in range(size)]
    return x


def setX2(begin, end, size):
    x = [1 / 2 * (begin - end) * np.cos(np.pi * (2 * i + 1) / (2 * size + 2) + (begin + end)) for i in range(size)]
    return x


def setF(f, x, n):
    for j in range(1, n):
        for i in range(j, n):
            f[i][j] = (f[i][j - 1] - f[i - 1][j - 1]) / (x[i] - x[i - j])
    return f


def setW(f, x, x_i, n):
    suma = 0.0
    for j in range(n-1):
        iloczyn = 1.0
        for i in range(j):
            iloczyn *= x_i - x[i]
        suma += f[j][j] * iloczyn
    return suma


def setY(x):
    y = [(1 / (1 + (el *     el))) for el in x]
    return y


if __name__ == '__main__':

    # dane
    for n in range(5, 25, 5):
        print(f"------------------------------------------ {n} ------------------------------------------")
        f = np.zeros((n, n))
        # x = setX2(-5, 5, n)
        x = setX(-5, 5, n)
        print("\nWektor x\n", x)
        y = setY(x)

        print("\nWektor y\n", y)
        for i in range(n):
            f[i][0] = y[i]
        f = setF(f, x, n)
        # print("\nMacierz f :\n", f)
        W = []
        for xi in np.arange(-5, 5.01, 0.1):
            W.append(setW(f, x, xi, n + 1))


        # ---------------------------------- CHARTS ------------------------------------------
        print("\nWielomian W(x):\n", W)
        xlabel = np.linspace(-5, 5.01, 101)
        plt.plot(xlabel, W, label=r"W(x)")
        plt.plot(x, y, label=r"f(x)")
        plt.xlabel(r"$x$")
        plt.ylabel("W(x) oraz f(x)")
        plt.title(r"Wykres W(x) oraz f(x) dla n = $%i$"% n)
        plt.legend()
        plt.show()
