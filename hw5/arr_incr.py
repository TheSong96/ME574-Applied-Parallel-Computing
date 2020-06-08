import numpy as np
from numba import cuda

@cuda.jit
def arradd(a, c):
    n = a.shape[0]
    i = cuda.grid(1)
    if i<n:
        a[i] = a[i] + c

def main():
    a = np.zeros(10)
    d_a = cuda.to_device(a)
    blocks, threads = 1, 32

    arradd[blocks, threads](d_a, 2.0)

    res = d_a.copy_to_host()
    print(res)

if __name__ == "__main__": 
    main()
