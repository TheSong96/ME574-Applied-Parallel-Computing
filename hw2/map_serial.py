import math
import numpy as np


def sFunc(x):
    return (1. - 2. * math.sin(np.pi * x)**2)


def sArray(x):
    n = x.size

    f = np.zeros(n)

    for i in range(n):
        f[i] = sFunc(x[i])

    return f