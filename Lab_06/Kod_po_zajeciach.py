import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


def Newton(F, J, r):
    k = 0
    Norms = []
    XY = []
    F_matrix = F(r[0], r[1])
    while True:
        XY.append(r)

        k += 1
        delta = - (np.linalg.inv(J(r[0], r[1])) @ F_matrix)
        r = r + delta
        F_matrix = F(r[0], r[1])
        Norms.append(np.linalg.norm(delta))
        if np.linalg.norm(delta) < 1e-6:
            break
    return r[0], r[1], k, Norms, XY


def F(x, y):
    return np.array([2 * x * pow(y, 2) - 3 * pow(x, 2) * y - 2, pow(x, 2) * pow(y, 3) + 2 * x * y - 12])


def J(x, y):
    return np.array([[2 * pow(y, 2) - 6 * x * y, 4 * x * y - 3 * pow(x, 2)],
                     [2 * x * pow(y, 3) + 2 * y, 3 * pow(x, 2) * pow(y, 2) + 2 * x]])


if __name__ == '__main__':
    print("---------------Dla r = [10, -4]---------------")
    # Dla r = [10, -4]
    r_0 = np.array([10, -4])
    print("Macierz F :\n", F(r_0[0], r_0[1]))
    print("Macierz J :\n", J(r_0[0], r_0[1]))
    x, y, k_1, norms_1, XY_1 = Newton(F, J, r_0)
    print("x, y, k: {} , {}, {}".format(x, y, k_1))
    print("---------------Dla r = [10, 10]---------------")
    # Dla r = [10, 10]
    r_0 = np.array([10, 10])
    print("Macierz F :\n", F(r_0[0], r_0[1]))
    print("Macierz J :\n", J(r_0[0], r_0[1]))
    x, y, k_2, norms_2, XY_2 = Newton(F, J, r_0)
    print("x, y, k: {} , {}, {}".format(x, y, k_2))

    # ---------------------------------- CHARTS ------------------------------------------

    X_1, Y_1 = map(list, zip(*XY_1))
    X_2, Y_2 = map(list, zip(*XY_2))

    plt.plot(norms_1, label=r"$k=%i$" % k_1, scaley="log")

    plt.xlabel(r"$k$")
    plt.ylabel(r"$||\Delta r||$")
    plt.legend()
    plt.title("Dla r = [10, -4]")
    plt.show()

    plt.plot(norms_2, label=r"$k=%i$" % k_2, scaley="log")

    plt.xlabel(r"$k$")
    plt.ylabel(r"$||\Delta r||$")
    plt.legend()
    plt.title("Dla r = [10, 10]")

    plt.show()
    # -------------------------------- X & Y CHART --------------------------------
    plt.scatter(X_1, Y_1)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Dla r = [10, -4]")
    plt.show()

    plt.scatter(X_2, Y_2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Dla r = [10, 10]")

    plt.show()
