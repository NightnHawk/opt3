import random
import numpy as np
from scipy.optimize import minimize

from methods import *

bounds = [(1., - np.sqrt(15)), (4., np.sqrt(15))]
randSet = []
for i in range(10):
    randSet.append([random.uniform(1., 4.), random.uniform(- np.sqrt(15), np.sqrt(15))])

for i in range(10):
    x_min = nelder_mead(objective_function, x0=randSet[i], s0=0.5, alpha=1., beta=0.5, gamma=2., delta=0.5, epsilon=0.5,
                        nMax=1000, bounds=bounds)
    print(x_min, ', ', objective_function(x_min)[0], '\n')

    x_min_penalty = penalty_function(func=objective_function, x0=randSet[i], penalty=S, penalty_c=1., alpha=1.,
                                     epsilon=0.5, nMax=1000, NM_s0=0.5, NM_alpha=1., NM_beta=0.5, NM_gamma=2.,
                                     NM_delta=0.5, NM_epsilon=0.5, NM_nMax=1000, bounds=bounds)
    print(x_min_penalty, ', ', objective_function(x_min_penalty), '\n')

    x_min_build_in = minimize(fun=objective_function, x0=np.array(randSet[i]), method='Nelder-Mead')
    print(x_min_build_in.x, ', ', x_min_build_in.fun, '\n\n')
