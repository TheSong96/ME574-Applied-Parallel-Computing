import numpy as np
import math
from numba import cuda


@cuda.jit(device=True)
def sFunc(x):
    return (1. - 2. * math.sin(np.pi * x)**2)


@cuda.jit
def sKernel(d_f, d_x):
    i = cuda.grid(1)
    n = d_x.size
    if i < n:
        d_f[i] = sFunc(d_x[i])


def sArray(x):
    n = x.size

    d_x = cuda.to_device(x)
    d_f = cuda.device_array(n, dtype=np.float64)

    TPB = 32
    gridDim = (n + TPB - 1) // TPB
    blockDim = TPB

    sKernel[gridDim, blockDim](d_f, d_x)
    return d_f.copy_to_host()
