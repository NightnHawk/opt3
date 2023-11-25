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
    x0 = np.asfarray(x0, dtype=np.double)  # make sure that all x0 values are stored as float64 type values
    simplex = np.empty((N + 1, N), dtype=np.double)  # prepare an array for simplex vertices to be stored
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


def penalty_function_2(func, x0, s0, c1, alpha, beta, gamma, delta, epsilon, nMax: int, bounds=None):
    x0 = np.atleast_1d(x0).flatten()
    x0 = np.asfarray(x0, dtype=np.double)
    s0 = np.double(s0)
    alpha = np.double(alpha)
    beta = np.double(beta)
    gamma = np.double(gamma)
    delta = np.double(delta)
    epsilon = np.double(epsilon)

    i = 1
    def F(x):
        func(x) + c1 * s0
    xi = nelder_mead(func=F, x0=x0, s0=s0, alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon, nMax=nMax, bounds=bounds)[0]
    c1 = alpha * c1

    while abs(xi - x0) < epsilon:
        x0 = xi
        xi = nelder_mead(func=F, x0=x0, s0=s0, alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon, nMax=nMax, bounds=bounds)[0]
        c1 = alpha * c1
    return xi
# def penalty_function(f, x0, c1, alpha, epsilon, Nmax: int, S: float):
#     i = 0
#     x = np.array([x0])
#     c = c1
#     while True:
#         def F(x):
#             f(x) + c * S
#         x = np.append(x, nelder_mead(F, x[i], S, alpha, 0.5, 2, 0.5, epsilon, Nmax)[0])
#         c = alpha * c
#         i += 1
#         if (i < 2):
#             if abs(x[i]) < epsilon: break
#         else:
#             if abs(x[i] - x[i - 1]) < epsilon: break
#     return x[i]

def penalty_function(f, S, x0, c0, alpha, epsilon, Nmax, bounds = None):
    x_prev = x0
    c = c0
    i = 0

    while True:
        i += 1

        # Wyznacz F(i)(x) = f(x) + c(i)S(x)
        F = lambda x: f(x) + c * S

        # Wyznacz x(i) dla F(i) startując z x(i-1)
        # Zakładamy, że mamy dostęp do jakiegoś algorytmu optymalizacji
        # który zwraca minimum funkcji F startując z punktu x_prev
        #x = optimize(F, x_prev)
        x = nelder_mead(F,x_prev,0.5 ,alpha,0.5,2,0.5,epsilon,Nmax, bounds=bounds)[0]

        # Aktualizuj c(i+1) = α·c(i)
        c *= alpha

        # Sprawdź warunek zakończenia
        if np.linalg.norm(x[0] - x_prev[0]) < epsilon:
            return x

        # Aktualizuj x_prev
        x_prev = x

        # Sprawdź, czy przekroczono maksymalną liczbę wywołań funkcji celu
        if i > Nmax:
            # raise Exception("Przekroczono maksymalną liczbę wywołań funkcji celu")
            return None

    return x