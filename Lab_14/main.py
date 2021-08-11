import random

import matplotlib.pyplot as plt
import numpy as np

# funkcja gestosci prawdopodobienstwa
fun = lambda x, mi, delta: -(abs(x - mi) / delta ** 2) + 1 / delta


def mi_delta(x):
    mi = (x.min() + x.max()) / 2
    delta = mi + x.max()
    return mi, delta


def distribution_function(x, mi, delta):
    if x <= mi:
        return ((-1 / delta ** 2) * (-(x ** 2 / 2) + mi * x) + (x / delta) - (
                (-1 / delta ** 2) * (- (mi - delta) ** 2 / 2 + mi * (mi - delta)) + (mi - delta) / delta))
    else:
        return ((-1 / delta ** 2) * (x ** 2 / 2 - mi * x) + x / delta - (
                -1 / delta ** 2 * (mi ** 2 / 2 - mi ** 2) + mi / delta) + 1 / 2)


def rozklad_jednorodny():
    # Dane
    x_0 = 10
    n = 10 ** 4
    a = 123
    c = 1
    m = 2 ** 15
    x = [x_0]

    for i in range(1, n):
        x.append((a * x[i - 1] + c) % m)

    x = np.array(x) / (m + 1.)

    # ---------------- CHART ----------------
    plt.figure()
    plt.plot(x[1::], x[0:n - 1:], "rs", markersize=1)
    plt.title(f'')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.yscale("log")
    plt.grid()
    plt.savefig('wykres1.png')
    plt.show()

    mi, delta = mi_delta(x)
    X_arr = []
    for i in range(1, n):
        X_arr.append(fun(x[i] / (m + 1.0), mi, delta))

    plt.hist(X_arr, color='b', edgecolor='w', bins=14)
    plt.savefig('wykres2.png')
    plt.show()


def rozklad_trojkatny():
    # Dane
    n = 10 ** 3
    mi = 4
    delta = 3
    k = 10
    start = mi - delta
    end = mi + delta
    h = (end - start) / k

    n_arr = np.zeros(k)
    x_arr = []

    # Generowanie n liczb o rozkladzie trojkatnym
    for i in range(n):
        e1 = random.uniform(0, 1)
        e2 = random.uniform(0, 1)
        x = mi + (e1 + e2 - 1) * delta
        x_arr.append(x)

    for x in x_arr:
        for j in range(k):
            if x < start + (j + 1) * h:
                n_arr[j] = n_arr[j] + 1
                break

    pi_val = [distribution_function(iter + h, mi, delta) - distribution_function(iter, mi, delta) for
              iter in np.arange(start, end, h)]

    plt.bar([i for i in np.arange(start, end, h)], n_arr / n, align='edge', width=0.4, color='sienna')
    plt.plot([i for i in np.arange(start, end, h)], pi_val, color='gold', label='$p_i$')
    plt.plot([i for i in np.arange(start, end, h)], pi_val, 'o', color='blue')
    plt.xlabel('$X$')
    plt.ylabel('$n_i/n$')
    plt.savefig('wykres3.png')
    plt.show()


if __name__ == '__main__':
    rozklad_jednorodny()
    rozklad_trojkatny()
