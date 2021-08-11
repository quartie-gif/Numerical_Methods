import math

import matplotlib.pyplot as plt
import numpy as np

# dane
a = 0
b = np.pi

fun = lambda x, m, k: x ** m * math.sin(k * x)


def sum(i, x, m, k):
    nominator = pow(k * x, 2 * i + m + 2)
    denominator = pow(k, m + 1) * math.factorial(2 * i + 1) * (2 * i + m + 2)
    return pow(-1, i) * (nominator / denominator)


def simpson_method(p_arr, m, k, a, b):
    I_arr = []
    for p in p_arr:
        h = (b - a) / p  # dzielimy na n jednakowych wezlow
        sum = 0
        for i in np.arange(a, b, h):
            leftSide = i
            middle = i + h / 2
            rightSide = i + h
            Integral = 1 / 3 * h / 2 * (1 * fun(leftSide, m, k) + 4 * fun(middle, m, k) + 1 * fun(rightSide, m, k))
            sum = sum + Integral

        I_arr.append(sum)
    return I_arr  # zwracamy liste wartości sum dla odpowiednich wezlow


if __name__ == '__main__':

    sum_arr = []
    arr_m_k = [(0, 1), (1, 1), (5, 5)]
    n_arr = [11, 21, 51, 101, 201]  # n = 2p + 1 = 11, 21, 51, 101, 201, zakres 0 do pi, wiec liczymy dla p
    numer = 0
    for m, k in arr_m_k:
        with open(f'wyniki_dla_m={m}_k={k}.txt', 'w') as file:
            suma = 0
            for i in range(30):
                suma += sum(i, b, m, k) - sum(i, a, m, k)
                file.write(str(suma) + '\n')
                sum_arr.append(suma)
                print(suma)

        simpson = simpson_method([(n-1)/2 for n in n_arr], m, k, a, b)
        I = sum_arr[-1]
        # print(I)
        # ---------------- CHART ----------------
        plt.figure()
        plt.plot([n for n in n_arr], [abs(C - I) for C in simpson], color='sienna')
        plt.title(f'Wykres zależności $|C-I|$ od ilości węzłów dla m={m} oraz k ={k}')
        plt.xlabel('n - ilosc wezlow')
        plt.ylabel('$|C-I|$')
        plt.yscale("log")
        plt.grid()
        plt.savefig('wykres_dla_m_%d.png' % m)
        plt.show()

        # plt.figure()
        # plt.plot([x for x in range(len(sum_arr))], sum_arr)
        # plt.title(f'Wykres dla metody rozwiniecia')
        # plt.xlabel('liczba wyrazów')
        # plt.ylabel('wartość I')
        # plt.grid()
        # plt.savefig(f'wykres_dla_metody_rozwiniecia_{numer}.png')
        # plt.show()
        # sum_arr.clear()
        # numer = numer + 1
