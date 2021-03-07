from pylab import *

data = open('data.txt', 'r')

t = []
wychylenie = []
h = 0.
for line in data:
    wychylenie.append(float(line.split()[1]))
    t.append(float(line.split()[0]))

print('\n', t, '\n')
print(wychylenie)
plt.plot(t, wychylenie)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.show()

data.close()
