import numpy as np
from numba import cuda


@cuda.jit(device=True)
def power_of_half(n):
    return 0.5**n


@cuda.jit()
def kernel(d_n):
    size = d_n.size
    i = cuda.grid(1)
    if i < size:
        epsilon = power_of_half(d_n[i])
        if 1. == (1.+epsilon):
            print(i, epsilon)

def wrapper(n):
    d_n = cuda.to_device(n)
    TPB = 32
    BPG = (n.size-1)//TPB + 1
    kernel[BPG, TPB](d_n)


def main():
    n = np.arange(101) # array of n
    wrapper(n)


if __name__ == '__main__':
    main()
