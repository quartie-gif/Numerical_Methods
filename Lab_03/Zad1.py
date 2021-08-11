import time

import matplotlib.pyplot as plt
import numpy as np

# Dane
N = 1000
m = 5
A = np.zeros((N, N), dtype=float)
x_1 = np.ones(N, dtype=float)
x_0 = np.zeros(N, dtype=float)
r_norm = []
x_norm = []
iter_arr = []

# inicjalizacja macierzy A

for i in range(N):
    for j in range(N):
        if (abs(i - j) <= m):
            A[i][j] = 1 / (1 + abs(i - j))
        else:
            A[i][j] = 0

# inicjalizacja macierzy b

b = np.zeros(N)
for i in range(N):
    b[i] = i


def grad(_A, _b, _x, _r_norm, _x_norm, _iter_arr, _file):
    _r = _b - _A @ _x
    k = 0
    while np.linalg.norm(_r) > 1e-6:
        Ar = _A @ _r
        alpha = (_r @ _r) / (_r @ Ar)
        _x = _x + alpha * _r
        _r = _r - alpha * Ar

        _iter_arr.append(k)
        _r_norm.append(np.linalg.norm(_r))
        _x_norm.append(np.linalg.norm(_x))

        _file.write("{} | {} | {} | {}\n".format(k, np.linalg.norm(_r), alpha, np.linalg.norm(_x)))

        k += 1

# Zapis danych do pliku tekstowego, gdy x wypelnione zerami

file = open("x_0.txt", "w")
t_start = time.time()
grad(A, b, x_0, r_norm, x_norm, iter_arr, file)
t_finish = time.time()
file.close()

print("Wektor x wypelniony zerami, czas : " + str(t_finish - t_start) + " sek")

# Zapis danych do pliku tekstowego, gdy x wypelnione jedynkami

file = open("x_1.txt", "w")
t_start = time.time()
grad(A, b, x_1, r_norm, x_norm, iter_arr, file)
t_finish = time.time()
file.close()

print("Wektor x wypelniony jedynkami, czas : " + str(t_finish - t_start) + " sek\n")

# wykresy

xlabel_1 = np.array(iter_arr)
ylabel_1 = np.array(x_norm)

plt.figure()
plt.xlabel("nr iteracji")
plt.ylabel("norma x")
plt.scatter(xlabel_1, ylabel_1, 12)
plt.show()

xlabel_2 = np.array(iter_arr)
ylabel_2 = np.array(r_norm)

plt.figure()
plt.xlabel("nr iteracji")
plt.ylabel("norma r")
plt.scatter(xlabel_2, ylabel_2, 5)
plt.yscale('log')
plt.show()

plt.show()
