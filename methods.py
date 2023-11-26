import sys
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
    return np.sin(np.pi * np.sqrt((x[0] / np.pi) ** 2 + (x[1] / np.pi) ** 2) / (
            np.pi * np.sqrt(((x[0] / np.pi) ** 2 + (x[1] / np.pi) ** 2))))


def nelder_mead(func, x0, s0, alpha, beta, gamma, delta, epsilon, nMax: int, bounds=None):
    f = functionWrapper(func)
    N = len(x0)  # calculate the dimension of our linear space
    e = np.eye(len(x0))  # generate basis vectors

    # cast all parameters to float type
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    delta = float(gamma)
    epsilon = float(epsilon)
    x0 = np.atleast_1d(x0).flatten()  # make sure that x0 is a flat array of at least one dimension
    x0 = np.asfarray(x0, dtype=np.float64)  # make sure that all x0 values are stored as float64 type values
    simplex = np.empty((N + 1, N), dtype=np.float64)  # prepare an array for simplex vertices to be stored
    if bounds is not None:
        lower_bound, upper_bound = bounds
        x0 = np.clip(x0, lower_bound, upper_bound)
        simplex[0] = np.clip(x0, lower_bound, upper_bound)
        for i in range(1, N):
            point = simplex[0] + s0 * e[i]
            simplex[i] = np.clip(point, lower_bound, upper_bound)
    else:
        simplex[0] = x0  # set first vertex as x0
        for i in range(1, N):
            simplex[i] = simplex[0] + s0 * e[i]
            # set rest of vertices but calculating the value moved by the length of s_init in the direction of basis vector e

    # find minimum and maximum amongst vertices of gotten simplex
    index_min = np.argmin(np.array([f.f(point) for point in simplex]))
    simplex_min = simplex[index_min]
    index_max = np.argmax(np.array([f.f(point) for point in simplex]))
    simplex_max = simplex[index_max]

    simplex_centroid = np.sum(np.delete(simplex, index_max)) / N  # calculate the centroid of the simplex
    simplex_reflection = simplex_centroid + alpha * (
            simplex_centroid - simplex_max)  # calculate the reflected vertex of the maximal simplex vertex
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
        difference = [np.linalg.norm(simplex_min - simplex[i]) for i in range(len(simplex) - 1)]
        difference_max = max(difference)
    return simplex_min, f.get_count()


def g1(x):
    if len(x) != 1:
        x = np.float64(x[0])
    return -x + 1


def g2(x):
    if len(x) != 1:
        x = np.float64(x[1])
    return -x + 1


a_array = [4., 4.4934, 5.]


def g3(x):
    return np.sqrt(x[0] ** 2 + x[1] ** 2 - a_array[0])


penalty_array = [g1, g2, g3]


def S(x, bounds):
    if bounds is not None:
        lower_bound, upper_bound = bounds
        if lower_bound[0] < x[0] < upper_bound[0] and lower_bound[1] < x[1] < upper_bound[1]:
            penalty = []
            temp = np.array([penalty_array[i](x) for i in range(3)])
            penalty.extend([(-1. / temp[i]) for i in range(3)])
            return sum(penalty)
        else:
            penalty = []
            temp = np.array([penalty_array[i](x) for i in range(3)])
            penalty.extend([max(0, temp[i]) for i in range(3)])
            for element in penalty:
                element *= element
            return sum(penalty)
    else:
        return 0.5


def penalty_function(func, x0, alpha=1., epsilon=0.5, penalty=S, penalty_c=1., nMax=1000, NM_s0=0.5, NM_alpha=1.,
                     NM_beta=0.5,
                     NM_gamma=2., NM_delta=0.5, NM_epsilon=0.5, NM_nMax=1000, bounds=None):
    x0 = np.atleast_1d(x0).flatten()
    x0 = np.asfarray(x0, dtype=np.double)

    NM_s0 = np.float64(NM_s0)
    NM_alpha = np.float64(NM_alpha)
    NM_beta = np.float64(NM_beta)
    NM_gamma = np.float64(NM_gamma)
    NM_delta = np.float64(NM_delta)
    NM_epsilon = np.float64(NM_epsilon)

    x_prev = x0
    c = penalty_c

    i = 0

    while True:
        i += 1
        if callable(penalty):
            F = lambda x: func(x) + c * penalty(x, bounds)
        else:
            F = lambda x: func(x) + c * penalty
        x = nelder_mead(func=F, x0=x_prev, s0=NM_s0, alpha=NM_alpha, beta=NM_beta, gamma=NM_gamma, delta=NM_delta,
                        epsilon=NM_epsilon, nMax=NM_nMax, bounds=bounds)[0]
        c *= alpha
        if np.linalg.norm(x - x_prev) < epsilon:
            return x
        x_prev = x
        if i > nMax:
            break
    return x
