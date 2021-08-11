import matplotlib.pyplot as plt
import numpy as np


# H matrix
def set_H(L, N):
    H = np.zeros((N, N))
    for i in range(N):
        x_i = -L + i * delta
        H[i][i] = (delta ** -2) + (x_i ** 2) / 2
        if i > 0:
            H[i - 1][i] = -1 / (2 * delta ** 2)
            H[i][i - 1] = -1 / (2 * delta ** 2)
    return H


# Householder
def householder(A):
    m, n = np.shape(A)
    R = np.copy(A)
    Q = np.identity(m)
    for k in range(n - 1):
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_k = np.identity(m)
        Q_k[k:, k:] -= 2.0 * np.outer(v, v)
        R = np.dot(Q_k, R)
        Q = np.dot(Q, Q_k)
    return Q, R


if __name__ == '__main__':
    # Data
    L = 5
    N = 50
    delta = 2 * L / N
    x = np.linspace(-L, L, N)
    # Setting matrix
    H = set_H(L, N)
    Q, R = householder(H)
    # printing matrices
    print('Q:', Q.round(2))
    print('R:', R.round(2))

    # ---------------------------------- CHART ------------------------------------------

    # w, v = np.linalg.eigh(H)
    #
    # print(w, v)

    P = np.eye(N)

    for i in range(N):
        Q, R = householder(H)
        P = P @ Q
        H = R @ Q

    for k in range(1, 5):
        plt.plot(x, P[:, -k] / np.linalg.norm(P[:, k]), label=r"$n=%i$" % k)

    plt.xlabel(r"$r$")
    plt.ylabel(r"$\psi(r)$")
    plt.legend()
    plt.show()
