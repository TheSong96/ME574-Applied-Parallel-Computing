import numpy as np
from numba import jit, cuda, float32, int32
import numba

# TPB = 128
# #define length of the shared memory array
# NSHARED = 130 #value must agree with TPB + 2*RAD

TPB = 16
NSHARED = 18

@cuda.jit(device = True)
def p(x0):
    return x0**2

@cuda.jit #Lazy compilation
#@cuda.jit('void(float32[:], float32[:])') #Eager compilation
def pKernel(d_f, d_x):
    i = cuda.grid(1)
    n = d_x.shape[0]
    if i < n:
        d_f[i] = p(d_x[i])

def pArray(x):
    n = x.shape[0]
    d_x = cuda.to_device(x)
    d_f = cuda.device_array(n, dtype = np.float32) #need dtype spec for eager compilation
    pKernel[(n+TPB-1)//TPB, TPB](d_f, d_x)
    return d_f.copy_to_host()

@cuda.jit
#@cuda.jit("void(float32[:],float32[:],float32[:])")
def deriv_kernel(d_deriv, d_f, d_stencil, rad):
    n = d_f.shape[0]
    i = cuda.grid(1)
    sh_f = cuda.shared.array(NSHARED, dtype = numba.float32) # elements will be initialized with 0.0
    # print("sh_f[0] = ", sh_f[4])
    #thread index (and index for optional shared output array)
    tIdx = cuda.threadIdx.x
    bIdx = cuda.blockIdx.x
    # print(tIdx, bIdx)
    # print("cuda.blockDim.x = ", cuda.blockDim.x)
    #index for shared input array
    shIdx = tIdx + rad

    if i>=n:
        return

    #Load regular cells
    sh_f[shIdx] = d_f[i]

    #Halo cells- Check that the entries to be loaded are within array bounds
    if tIdx < rad:
        if i >= rad:
            sh_f[shIdx - rad] = d_f[i-rad]
        if i + cuda.blockDim.x < n:
            sh_f[shIdx + cuda.blockDim.x] = d_f[i + cuda.blockDim.x]

    #make sure that shared array is fully loaded before any thread reads from it
    cuda.syncthreads()


    #write values only where the full stencil is "in bounds"
    if i >= rad and i < n-rad:
        stencil_dot =  sh_f[shIdx] * d_stencil[rad]
        for d in range(1,rad+1):
            stencil_dot += sh_f[shIdx-d]*d_stencil[rad-d] + sh_f[shIdx+d]*d_stencil[rad+d]
        d_deriv[i] = stencil_dot

def nth_deriv_shared(f, order, rad):
    n = f.shape[0]
    if rad == 1:
        if order == 1:
            stencil =(n-1)/2. * np.array([-1., 0., 1.])
        elif order == 2:
            stencil = (n-1)*(n-1)* np.array([1., -2., 1.])
    elif rad == 2:
        if order == 1:
            stencil =(n-1)/12. * np.array([1., -8., 0., 8., -1.])
        elif order == 2:
            stencil = (n-1)*(n-1)* np.array([-1., 16., -30., 16., -1.])/12.
    print(order, stencil)
    d_f = cuda.to_device(f)
    d_stencil = cuda.to_device(stencil)
    d_deriv = cuda.device_array(n, dtype = np.float32)
    deriv_kernel[(n+TPB-1)//TPB, TPB](d_deriv, d_f, d_stencil, rad)

    return d_deriv.copy_to_host()

def nth_deriv_serial(f, order):
    n = f.shape[0]
    if order == 1:
        stencil = np.array([-1., 0., 1.])
        c = (n-1) / 2.
    elif order == 2:
        stencil = np.array([1., -2., 1.])
        c = (n-1)*(n-1)
    deriv = np.zeros(n)

    for i in range(1,n-1):
        deriv[i] = c * (
            f[i-1]*stencil[0]+
            f[i] * stencil[1]+
            f[i+1]*stencil[2])

    return deriv

@cuda.jit
def df_kernel(d_deriv, d_f, stencil):
    i = cuda.grid(1)
    n = d_f.shape[0]
    if i >= 1 and i < n-1:
        d_deriv[i] = (
            d_f[i-1]*stencil[0]+
            d_f[i] * stencil[1]+
            d_f[i+1]*stencil[2])

def nth_deriv_parallel(f, order):
    n = f.shape[0]
    if order == 1:
        stencil =(n-1)/2. * np.array([-1., 0., 1.])
    elif order == 2:
        stencil = (n-1)*(n-1)* np.array([1., -2., 1.])
    d_f = cuda.to_device(f)
    d_deriv = cuda.device_array(n, dtype = np.float32)
    df_kernel[(n+TPB-1)//TPB, TPB](d_deriv, d_f, stencil)

    return d_deriv.copy_to_host()
