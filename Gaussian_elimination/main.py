import numpy as np


def gauss_solver(_a, _b):
    n = len(b)

    # Forward Elimination
    #   a_ij = a_ij - a_ik/a_kk * a_kj
    # b_i = b_i - a_ik/a_kk * a_kj
    # k : diagonals -- 0 to n-2
    # i : rows -- k+1 to n-1
    # j : columns -- k to n-1

    for k in range(0, n - 1):
        for i in range(k + 1, n):
            ratio = _a[i, k] / _a[k, k]
            for j in range(k, n - 1):
                _a[i, j] -= a[k, j] * ratio
                _b -= ratio * _b[k]

    # Back substitution
    # x_n-1 = b_n-1/a_n-1,n-1
    # x_i = (b_i - sum_j (a_ij * x_j ))/ a_ii
    # i : n-2 to 0
    # j : i+1  to  n-1
    x = np.zeros(n)
    x[n-1] = b[n-1]/_a[n-1,n-1]
    for i in range(n-2, -1, -1):
        sum_j = 0
        for j in range (i+1, n):
            sum_j += _a[i,j]*x[j]
        x[i] = (_b[i] - sum_j)/_a[i,i]
    print("residual", np.linalg.norm(np.dot(_a, x) - _b))
    print("Printing the soulution using our algorithm : ")
    return x


a = np.array([[3., -2., 2.],
              [2., 3., 14.],
              [3., 3., 5.]], float)
b = np.array([1, 2, 3], float)

print(gauss_solver(a, b))  # printing our solution

print("Solution using numpy: ", np.linalg.solve(a,b))


# residual : | A*x = b|

