import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

import methods

"""
Data
"""
m = 0.6
r = 0.12
g = 9.81
C = 0.47
rho = 1.2

def equations_of_motion(state, t, omega):
    x, dxdt, y, dydt = state

    S = np.pi * r**2
    D_x = 0.5 * C * rho * S * dxdt**2
    D_y = 0.5 * C * rho * S * dydt**2
    F_M_x = rho * dydt * omega * np.pi * r**3
    F_M_y = rho * dxdt * omega * np.pi * r**3

    dx2dt2 = (-D_x -F_M_x) / m
    dy2dt2 = (-D_y -F_M_y - m*g) / m

    return dxdt, dx2dt2, dydt, dy2dt2

def simulate_ball_drop(v0x, omega):
    initial_state = [v0x, 0., 50., 0.]
    t = np.arange(0., 7., 0.01)

    solution = odeint(equations_of_motion, initial_state, t, args=(omega,))
    x_final = solution[-1, 0]
    return x_final

def objective_function(params):
    v0x, omega = params
    return -simulate_ball_drop(v0x, omega)

initial_guess = [2.23, -15.1]  # Adjusted initial values for v0x and omega

constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] - (-10)},
    {'type': 'ineq', 'fun': lambda x: 10 - x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1] - (-23)},
    {'type': 'ineq', 'fun': lambda x: 23 - x[1]},
    {'type': 'ineq', 'fun': lambda x: simulate_ball_drop(x[0], x[1]) - 4},
    {'type': 'ineq', 'fun': lambda x: 6 - simulate_ball_drop(x[0], x[1])}
]

result = minimize(objective_function, initial_guess, constraints=constraints)
result1 = methods.nelder_mead(func=objective_function, x0=initial_guess, s0=0.5, alpha=1., beta=0.5, gamma=2., delta=0.5, epsilon=0.5, nMax=1000, bounds=None)
result2 = methods.penalty_function(func=objective_function, x0=initial_guess, alpha=1., epsilon=0.5, penalty=methods.S, penalty_c=1., nMax=1000, NM_s0=0.5, NM_alpha=1., NM_beta=0.5, NM_gamma=2., NM_delta=0.5, NM_epsilon=0.5, NM_nMax=1000, bounds=None)

print("Optimal solution:")
print("v0x =", result.x[0])
print('Nelder Mead: v0x =', result1[0][0])
print('Penalty: v0x =', result2[0])
print('Omega =', result.x[1])
print('Nelder Mead: omega =', result1[0][1])
print('Penalty: omega =', result2[1])

