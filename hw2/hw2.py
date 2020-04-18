import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from time import time


def gpu_total_memory():
    '''
    Query the GPU's properties via Numba to obtain the total memory of the device.
    '''
    # print(cuda.detect())
    c_context = cuda.current_context(devnum=None)
    # print(c_context.get_memory_info())
    return int(c_context.get_memory_info()[1])


def gpu_compute_capability():
    '''
    Query the GPU's properties via Numba to obtain the compute capability of the device.
    '''
    c_device = cuda.get_current_device()

    return c_device.compute_capability


def gpu_name():
    '''
    Query the GPU's properties via Numba to obtain the name of the device.
    '''
    c_device = cuda.get_current_device()

    return c_device.name


def max_float64s():
    '''
    Compute the maximum number of 64-bit floats that can be stored on the GPU
    '''
    return int(gpu_total_memory() / 16)


def map_64():
    '''
    Execute the map app modified to use 64-bit floats
    '''
    from map_parallel import sArray
    N = 64
    # while(True):
    x = np.linspace(0, 1, N, dtype=np.float64)
    y = sArray(x)
    print("Successfully run, when N = {}^{}".format(2, int(np.log2(N))))
    N = N * 2
    # plt.plot(x, y)
    # plt.savefig("cos(2pix).png")


@cuda.jit(device=True)
def f(x, r):
    '''
    Execute 1 iteration of the logistic map
    '''
    return r * x * (1 - x)


@cuda.jit
def logistic_map_kernel(ss, r, x, transient, steady):
    '''
    Kernel for parallel iteration of logistic map

    Arguments:
        ss: 2D numpy device array to store steady state iterates for each value of r
        r: 1D  numpy device array of parameter values
        x: float initial value
        transient: int number of iterations before storing results
        steady: int number of iterations to store
    '''
    i = cuda.grid(1)
    if i < r.size:
        x_old = x
        for j in range(transient):
            x_new = f(x_old, r[i])
            x_old = x_new
        for j in range(steady):  #iterate over the desired sequence
            ss[j, i] = x_old
            x_new = f(x_old, r[i])
            x_old = x_new


def parallel_logistic_map(r, x, transient, steady):
    '''
    Parallel iteration of the logistic map

    Arguments:
        r: 1D numpy array of float64 parameter values
        x: float initial value
        transient: int number of iterations before storing results
        steady: int number of iterations to store
    Return:
        2D numpy array of steady iterates for each entry in r
    '''
    d_r = cuda.to_device(r)
    print("r.size = ", r.size)
    d_ss = cuda.device_array((steady, r.size), dtype=np.float64)

    TPB = 32
    gridDim = (r.size + TPB - 1) // TPB
    blockDim = TPB

    logistic_map_kernel[gridDim, blockDim](d_ss, d_r, x, transient, steady)

    return d_ss.copy_to_host().transpose()


@cuda.jit(device=True)
def iteration_count(cx, cy, dist, itrs):
    '''
    Computed number of Mandelbrot iterations

    Arguments:
        cx, cy: float64 parameter values
        dist: float64 escape threshold
        itrs: int iteration count limit
    '''
    z_real = z_img = 0

    for i in range(itrs):
        z_real, z_img = z_real**2 - z_img**2 + cx, 2 * z_real * z_img + cy
        if (z_real**2 + z_img**2 > dist):
            return i
    return itrs


@cuda.jit
def mandelbrot_kernel(out, cx, cy, dist, itrs):
    '''
    Kernel for parallel computation of Mandelbrot iteration counts

    Arguments:
        out: 2D numpy device array for storing computed iteration counts
        cx, cy: 1D numpy device arrays of parameter values
        dist: float64 escape threshold
        itrs: int iteration count limit
    '''
    i, j = cuda.grid(2)
    nx, ny = out.shape
    if i < nx and j < ny:
        out[i, j] = iteration_count(cx[i], cy[j], dist, itrs)


def parallel_mandelbrot(cx, cy, dist, itrs):
    '''
    Parallel computation of Mandelbrot iteration counts

    Arguments:
        cx, cy: 1D numpy arrays of parameter values
        dist: float64 escape threshold
        itrs: int iteration count limit
    Return:
        2D numpy array of iteration counts
    '''
    nx = cx.size
    ny = cy.size

    d_x = cuda.to_device(cx)
    d_y = cuda.to_device(cy)
    d_f = cuda.device_array((nx, ny), dtype=np.float32)

    TPBX = TPBY = 32
    gridDims = ((nx + TPBX - 1) // TPBX, (ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)
    mandelbrot_kernel[gridDims, blockDims](d_f, d_x, d_y, dist, itrs)

    return d_f.copy_to_host()


if __name__ == "__main__":

    #Problem 1
    print("GPU memory in GB: ", gpu_total_memory() / 1024**3)
    print("Compute capability (Major, Minor): ", gpu_compute_capability())
    print("GPU Model Name: ", gpu_name())
    print("Max float64 count: ", max_float64s())

    #PASTE YOUR OUTPUT HERE#
    """
        GPU memory in GB:  7.92730712890625
        Compute capability (Major, Minor):  (6, 1)
        GPU Model Name:  b'GeForce GTX 1080'
        Max float64 count:  531992576
    """

    #Problem 2
    map_64()

    #PASTE YOUR ERROR MESSAGES HERE#
    """
        Max N = 2^29
        ERROR MESSAGES: numba.cuda.cudadrv.driver.CudaAPIError: [2] Call to cuMemAlloc results in CUDA_ERROR_OUT_OF_MEMORY
    """

    #Problem 3
    m, rmin, rmax = 128, 2.5, 4.0
    r = np.linspace(rmin, rmax, m)
    x, transient, steady = 0.5, 100, 1024
    ss = parallel_logistic_map(r, x, transient, steady)

    fig = plt.figure(3)
    plt.plot(r, ss, 'b.')
    plt.axis([rmin, rmax, 0, 1])
    plt.xlabel('r value')
    plt.ylabel('x value')
    plt.title('Iterations of the logistic map')
    plt.savefig("logistics map.png")

    #Problem 4

    width = height = 512
    real_low = imag_low = -2
    imag_high = real_high = 2
    real_vals = np.linspace(real_low, real_high, width)
    imag_vals = np.linspace(imag_low, imag_high, height)

    t1 = time()
    mandel = parallel_mandelbrot(real_vals, imag_vals, 2.5, 256)
    t2 = time()
    mandel_time = t2 - t1

    fig = plt.figure(4)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.colorbar()
    plt.savefig('mandelbrot.png')
    # plt.show()

    # boolean_mandel = mandel > 1
    # plt.imshow(boolean_mandel, extent=(-2, 2, -2, 2))
    # plt.savefig('boolean_mandel.png')
    # plt.show()

    print('It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time))
