import numpy as np
import matplotlib.pyplot as plt


def muller(x,y):
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    value = 0.
    for j in range(4):
        value += AA[j] * np.exp(aa[j] * (x - XX[j])**2 + \
            bb[j] * (x - XX[j]) * (y - YY[j]) + \
            cc[j] * (y - YY[j])**2)
    return value

X = np.linspace(-1.75,1.,300)
Y = np.linspace(-0.5,2.5,300)
xx,yy = np.meshgrid(X,Y)
zz = muller(xx,yy)
print(xx,yy,zz)
ax = plt.contourf(xx,yy.clip(max = 200),zz,40)
plt.legend()
plt.show()
