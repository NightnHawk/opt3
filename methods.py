# def penalty(x, c:float, alpha:float, epsilon:float, nMax:int):
#     i = 0;
#     while abs(x - x1) < epsilon:
#         i += 1
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

def nelder_mead(f, x, s_init, alpha:float, beta:float, gamma:float, delta:float, epsilon:float, nMax:int):
    fun = functionWrapper(f)
    e = np.eye(len(x))
    p = np.array(len(x) + 1)
    simplexSet = np.empty(len(x) + 1)
    pMin = None
    pMax = None
    p[0] = x
    for i in range(1, len(x)):
        p[i] = p[0] + s_init * e[i]
    for i in range(0, len(x) + 1):
        simplexSet[i] = fun.function(p[i])
    simplexSorted = simplexSet.copy().sort()
    minimum = simplexSorted[0]
    maximum = simplexSorted[-1]
    indexMin = np.where(simplexSet == minimum)[0]
    pMin = p[indexMin]
    indexMax = np.where(simplexSet == maximum)[0]
    pMax = p[indexMax]
    p_ = (sum(simplexSet) - pMax) / len(x)
    pOdb = p_ + alpha * (p_ - pMax)
    if fun.function(pOdb) < fun.function(pMin):
        pe = p_ + gamma * (pOdb - p_)   #pe p z ekspansji
        if fun.function(pe) < fun.function(pOdb):
            pMax = pe
        else:
            pMax = pOdb
    else:
        if fun.function(pMin) <= fun.function(pOdb) < f(pMax):
            pMax = pOdb
        else:
            pz = p_ + beta * (pMax - p_)
            if fun.function(pz) >= fun.function(pMax):
                for i in range(len(x)):
                    if i != indexMin:
                        p[i] = delta * (p[i] + pMin)
            else:
                pMax = pz
    if fun.get_count() > nMax:
        raise ValueError

    return pMin
