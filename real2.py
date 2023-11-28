import numpy as np
import math
import methods
from scipy.optimize import minimize

m = 0.6
r = 0.12
y0 = 100
vy0 = 0
x0 = 0
ro = 1.2
C = 0.47
S = math.pi * r ** 2
g = 9.81

t0 = 0
tk = 7
dt = 0.01

a = 0.5 * C * ro * S
b = ro * math.pi * r ** 3
l0 = 1
vx0 = 0


def d2xdt2(vx, vy, l, t):
    return -(a * vx ** 2 + b * l * vy) / m


def d2ydt2(vx, vy, l, t):
    return -(m * g + a * vy ** 2 + b * l * vx) / m


# ax0 = d2xdt2(vx0, vy0, l0, t0)
# ay0 = d2ydt2(vx0, vy0, l0, t0)
ax0 = 0
ay0 = -g


def Euler(u: []):
    f1 = d2xdt2
    f2 = d2ydt2
    t = np.array([i * dt for i in range(t0, int(tk / dt))])
    li = u[1]
    n = t.size
    vy = np.zeros(n)
    vx = np.zeros(n)
    ax = np.zeros(n)
    ay = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)
    vy[0] = vy0
    vx[0] = vx0
    ax[0] = ax0
    ay[0] = ay0
    x[0] = u[0]
    y[0] = y0
    for i in range(n - 1):
        ax[i + 1] = ax[i] + dt * f1(vx0, vy[i], li, t[i])
        ay[i + 1] = ay[i] + dt * f2(vx0, vy[i], li, t[i])
        vy[i + 1] = vy[i] + ay[i] * dt
        vx[i + 1] = vx[i] + ax[i] * dt
        x[i + 1] = x[i] + vx[i] * dt + (ax[i] * dt ** 2) / 2
        y[i + 1] = y[i] + vy[i] * dt + ay[i] * (dt ** 2) / 2
        if y[i + 1] <= 50 and x[i + 1] > 4.6: return [None], [None]
        if y[i + 1] < 0: return x[0:i+1], y[0:i+1]
    return x, y,


t = np.array([i * dt for i in range(t0, int(tk / dt))])
p = np.array([])
l = np.array([-23 + i * 0.1 for i in range(0, int(46 / 0.1))])
xw = np.array([-10 + i * 0.1 for i in range(0, int(20 / 0.1))])
xc = np.array([])
lc = np.array([])
for i in l:
    for j in xw:
        pom = np.array(Euler([j, i]))
        if pom.any(None):
            p = np.append(p, pom[0][-1])
            xc = np.append(xc, i)
            lc = np.append(lc, j)
print(max(p))
print(xc[np.where(p == max(p))])
print(lc[np.where(p == max(p))])
print(p.size)
print(xc.size)
print(lc.size)

# initial_guess = [2.23, -15.1]
# result1 = methods.nelder_mead(func=Euler, x0=initial_guess, s0=0.5, alpha=1., beta=0.5, gamma=2., delta=0.5,
#                               epsilon=0.5, nMax=1000, bounds=None)
#result2 = methods.penalty_function(func=Euler, x0=initial_guess, alpha=1., epsilon=0.5, penalty=methods.S, penalty_c=1., nMax=1000, NM_s0=0.5, NM_alpha=1., NM_beta=0.5, NM_gamma=2., NM_delta=0.5, NM_epsilon=0.5, NM_nMax=1000, bounds=None)


# print(result1)
# print(Euler(result1[0]))
# print()
# print(result2)
# print(Euler(result2[0]))
