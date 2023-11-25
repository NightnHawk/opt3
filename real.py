
"""
Data
"""
import numpy as np

m = 0.6
r = 0.12
y0 = 100
v0y = 0.0
x0 = 0.0
g = 9.81
C = 0.47
ro = 1.2
t0 = 0.
dt = 0.01
tk = 7.

def S(r):
    return np.pi * r**2

def Di(v):
    return 0.5 * C * ro * S(r) * v**2

def Fmi(v, omega):
    return ro * v * omega * np.pi * r**3