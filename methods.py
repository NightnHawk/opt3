# def penalty(x, c:float, alpha:float, epsilon:float, nMax:int):
#     i = 0;
#     while abs(x - x1) < epsilon:
#         i += 1
import sys
import warnings

import numpy as np


class functionWrapper:
    def __init__(self, f):
        self.function = f
        self.counter = 0

    def f(self, x):
        self.counter += 1
        return self.function(x)

    def f_df(self, x):
        return self.function(x)

    def get_count(self):
        return self.counter

    def reset_count(self):
        self.counter = 0


def objective_function(x):
    return np.sin(np.pi * np.sqrt((x[0]/np.pi)**2 + (x[1]/np.pi)**2)/(np.pi * np.sqrt(((x[0]/np.pi)**2 + (x[1]/np.pi)**2))))

def nelder_mead(func, x0, s0, alpha: float, beta: float, gamma: float, delta: float, epsilon: float, nMax: int, bounds=None):
    f = functionWrapper(func)
    N = len(x0)
    e = np.eye(len(x0))
    x0 = np.atleast_1d(x0).flatten()  # make sure that x0 is a flat array of at least one dimension
    x0 = np.asfarray(x0, dtype=np.double)  # make sure that all x0 values are stored as float64 type values
    simplex = np.empty((N + 1, N), dtype=np.double)  # prepare an array for simplex vertices to be stored
    if bounds is not None:
        lower_bound, upper_bound = bounds
        simplex[0] = np.clip(x0, lower_bound, upper_bound)
    else:
        simplex[0] = x0  # set first vertex as x0
    if bounds is not None:
        x0 = np.clip(x0, lower_bound, upper_bound)
        for i in range(1, N):
            point = simplex[0] + s0 * e[i]
            simplex[i] = np.clip(point, lower_bound, upper_bound)
            # set rest of vertices but calculating the value moved my the length of s_init in the direction of basis vector e
    else:
        for i in range(1, N):
            simplex[i] = simplex[0] + s0 * e[i]

    index_min = np.argmin(np.array([f.f(point) for point in simplex]))
    simplex_min = simplex[index_min]
    index_max = np.argmax(np.array([f.f(point) for point in simplex]))
    simplex_max = simplex[index_max]

    simplex_centroid = np.sum(np.delete(simplex, index_max)) / N
    simplex_reflection = simplex_centroid + alpha * (simplex_centroid - simplex_max)
    if f.f(simplex_reflection) < f.f(simplex_min):
        simplex_expansion = simplex_centroid + gamma * (simplex_reflection - simplex_centroid)

        if bounds is not None:
            simplex_expansion = np.clip(simplex_expansion, lower_bound, upper_bound)

        if f.f(simplex_expansion) < f.f(simplex_reflection):
            simplex_max = simplex_expansion
        else:
            simplex_max = simplex_reflection
    else:
        if f.f(simplex_min) <= f.f(simplex_reflection) < f.f(simplex_max):
            simplex_max = simplex_reflection
        else:
            simplex_contraction = simplex_centroid + beta * (simplex_max - simplex_centroid)
            if f.f(simplex_contraction) >= f.f(simplex_max):
                for i in range(N):
                    if i != index_min:
                        simplex[i] = delta * (simplex[i] + simplex_min)
            else:
                simplex_max = simplex_contraction

    if bounds is not None:
        simplex_max = np.clip(simplex_max, lower_bound, upper_bound)

    if f.get_count() > nMax:
        raise ValueError('Function call count exceeded')

    difference_max = sys.float_info.max
    while difference_max < epsilon:
        index_min = np.argmin(np.array([f.f(point) for point in simplex]))
        simplex_min = simplex[index_min]
        index_max = np.argmax(np.array([f.f(point) for point in simplex]))
        simplex_max = simplex[index_max]

        simplex_centroid = np.sum(np.delete(simplex, index_max)) / N
        simplex_reflection = simplex_centroid + alpha * (simplex_centroid - simplex_max)

        if bounds is not None:
            simplex_reflection = np.clip(simplex_reflection, lower_bound, upper_bound)

        if f.f(simplex_reflection) < f.f(simplex_min):
            simplex_expansion = simplex_centroid + gamma * (simplex_reflection - simplex_centroid)

            if bounds is not None:
                simplex_expansion = np.clip(simplex_expansion, lower_bound, upper_bound)

            if f.f(simplex_expansion) < f.f(simplex_reflection):
                simplex_max = simplex_expansion
            else:
                simplex_max = simplex_reflection

            if bounds is not None:
                simplex_max = np.clip(simplex_max, lower_bound, upper_bound)

        else:
            if f.f(simplex_min) <= f.f(simplex_reflection) < f.f(simplex_max):
                simplex_max = simplex_reflection

                if bounds is not None:
                    simplex_max = np.clip(simplex_max, lower_bound, upper_bound)

            else:
                simplex_contraction = simplex_centroid + beta * (simplex_max - simplex_centroid)

                if bounds is not None:
                    simplex_contraction = np.clip(simplex_contraction, lower_bound, upper_bound)

                if f.f(simplex_contraction) >= f.f(simplex_max):
                    for i in range(N):
                        if i != index_min:
                            simplex[i] = delta * (simplex[i] + simplex_min)
                    simplex = np.clip(simplex, lower_bound, upper_bound)
                else:
                    simplex_max = simplex_contraction

                    if bounds is not None:
                        simplex_max = np.clip(simplex_max, lower_bound, upper_bound)

        if f.get_count() > nMax:
            raise ValueError('Function call count exceeded')

        difference = [simplex_min - simplex[i] for i in range(len(simplex) - 1)]
        difference_max = max(difference)
    return simplex_min, f.get_count()