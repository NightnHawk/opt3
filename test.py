import random
import numpy as np
from scipy.optimize import minimize

from methods import nelder_mead, objective_function

a_array = [4., 4.4934, 5.]


def g1(x1):
    return -x1 + 1  # x1 >= 1


def g2(x2):
    return -x2 + 1  # x2 >= 1


def g3(x1, x2, a):
    return np.sqrt(x1 ** 2 + x2 ** 2) - a  # g3(x1,x2) <=0


bounds = [(1., - np.sqrt(15)), (4., np.sqrt(15))]
randSet = []
for i in range(10):
    randSet.append([random.uniform(1., 4.), random.uniform(- np.sqrt(15), np.sqrt(15))])

for i in range(10):
    x_min = nelder_mead(objective_function, x0=randSet[i], s0=0.5, alpha=1., beta=0.5, gamma=2., delta=0.5, epsilon=0.5,
                        nMax=1000, bounds=bounds)
    print(x_min, ', ', objective_function(x_min)[0])
    x_min_build_in = minimize(fun=objective_function, x0=np.array(randSet[i]), method='Nelder-Mead')
    print(x_min_build_in.x, ', ', x_min_build_in.fun)
