import time
import matplotlib.pyplot as plt
import numpy as np

# Dane
N = 1000
m = 5
A = np.zeros((N, N), dtype=float)
x = np.zeros(N, dtype=float)
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


# Funkcja implementujaca metode sprzezonych gradientow dla macierzy wstegowej
def grad(_A, _b, _x, _r_norm, _x_norm, _iter_arr, _file):
    _r = _b
    k = 0
    while np.linalg.norm(_r) > 1e-10:
        if k == 0:
            _r_temp = _r
        else:
            beta = - (_r_temp @ _A @ _r) / (_r_temp @ _A @ _r_temp)
            _r_temp = _r + beta * _r_temp
        alpha = (_r_temp @ _r) / (_r_temp @ _A @ _r_temp)
        _x = _x + alpha * _r_temp
        _r = _r - alpha * (_A @ _r_temp)

        _iter_arr.append(k)
        _r_norm.append(np.linalg.norm(_r))
        _x_norm.append(np.linalg.norm(_x))

        _file.write("{} | {} | {} | {}\n".format(k, np.linalg.norm(_r), alpha, np.linalg.norm(_x)))
        k += 1

# Zapis danych do pliku tekstowego

file = open("x(k).txt", "w")
t_start = time.time()
grad(A, b, x, r_norm, x_norm, iter_arr, file)
t_finish = time.time()
file.close()

print("sprzezony gradient czas : " + str(t_finish - t_start) + " sek")

# wykresy

xlabel = np.array(iter_arr)
ylabel = np.array(r_norm)

plt.figure()
plt.xlabel("nr iteracji")
plt.ylabel("norma r")
plt.yscale("log")
plt.scatter(xlabel, ylabel, 12)
plt.show()
