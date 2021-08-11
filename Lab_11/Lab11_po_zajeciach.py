import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft

np.seterr(divide='ignore', invalid='ignore')

f0 = lambda t: np.sin(1 * omega * t) + np.sin(2 * omega * t) + np.sin(3 * omega * t)


def set_t(dt, t_max):
    t = [i for i in np.arange(0, t_max, dt)]
    return t


def f_0_set(t):
    f_0 = np.zeros(shape=(len(t)))
    for i in range(len(t)):
        f_0[i] = f0(t[i])
    return f_0


def f_set(f_0):
    f = np.zeros(len(f_0))
    for i in range(len(f_0)):
        f[i] = f_0[i] + random.uniform(-.5, .5)
    return f


def g_set(t, sigma):
    g = lambda t: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1) * (t ** 2) / (2 * sigma * sigma))
    g1 = []
    g2 = []
    for i in range(len(t)):
        g1.append(g(t[i]))
        g2.append(g(-t[i]))
    return g1, g2


if __name__ == '__main__':
    k_list = [8, 10, 12]
    for k in k_list:

        # Dane poczatkowe
        N_k = 2 ** k
        T = 1.0
        t_max = 3 * T
        dt = t_max / N_k
        sigma = T / 20
        omega = 2 * np.pi / T

        # settings
        t = set_t(dt, t_max)

        f_0 = f_0_set(t)
        g_1, g_2 = g_set(t, sigma)

        g_1 = np.array(g_1)
        g_2 = np.array(g_2)

        # operations
        f = f_set(f_0)
        g = g_1 + g_2

        f_fourier = numpy.fft.fft(f)
        g_fourier = numpy.fft.fft(g)

        fg_fourier = np.multiply(f_fourier, g_fourier)
        fg_inverseFourier = numpy.fft.ifft(fg_fourier)

        # element o maksymalnym module
        f_max = np.amax(np.abs(fg_inverseFourier))

        # znormalizowana funkcja odszumiona
        fg_inverseFourier = fg_inverseFourier * 2.5 / f_max

        # -------------------------- FILE SAVE --------------------------

        with open('wyniki_dla_k_%d.txt' % k, 'w') as file:
            file.write("Sygnał niezaburzony:\n")
            file.write(str(f_0))
            file.write("\nZnormalizowany splot:\n")
            file.write(str(fg_inverseFourier))
        # -------------------------- CHART --------------------------
        plt.plot(t, f, label='Sygnał zaszumiony', color='silver')
        plt.plot(t, f_0, '--', label='Sygnał oryginalny', color='sienna')
        plt.plot(t, fg_inverseFourier, label='Sygnał odszumiony', color='tomato')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.legend()
        plt.grid()
        plt.savefig('wykres_dla_k_%d.png' % k, bbox_inches='tight')
        plt.show()
