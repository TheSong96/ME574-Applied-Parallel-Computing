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

def f_serial(x, r):
    '''
    Execute 1 iteration of the logistic map
    '''
    return r * x * (1 - x)

def logisticSteadyArray(x0,r,n_transient, n_ss):
    '''
    Conpute an array of iterates of the logistic map f(x)=r*x*(1-x)
    
    Inputs:
        x0: float initial value
        r: float parameter value
        n_transient: int number of initial iterates to NOT store
        n_ss: int number of iterates to store
        
    Returns:
        x: numpy array of n float64 values
    '''
    #create an array to hold n elements (each a 64 bit float)
    x = np.zeros(n_ss, dtype=np.float64) 
    x_old = x0 #assign the initial value
    for i in range(n_transient):
        x_new = f_serial(x_old, r)
        x_old = x_new
    for i in range(n_ss): #iterate over the desired sequence
        x[i] = x_old
        x_new = f_serial(x_old, r) #compute the output value and assign to variable x_new
        x_old = x_new #assign the new (output) value top be the old (input) value for the next iterate
    return x


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

    TPBX = TPBY = 2**4
    gridDims = ((nx + TPBX - 1) // TPBX, (ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)
    mandelbrot_kernel[gridDims, blockDims](d_f, d_x, d_y, dist, itrs)

    return d_f.copy_to_host()


if __name__ == "__main__":
    Prob1 = True
    Prob2 = True
    Prob3 = True
    Prob4 = True

    # Prob1 = Prob2 = Prob3 = False

    #Problem 1
    if Prob1:
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
    if Prob2:
        map_64()    

    #PASTE YOUR ERROR MESSAGES HERE#
    """
        Max N = 2^28
        ERROR MESSAGES: numba.cuda.cudadrv.driver.CudaAPIError: [2] Call to cuMemAlloc results in CUDA_ERROR_OUT_OF_MEMORY
    """

    #Problem 3
    if Prob3:
        m, rmin, rmax = 2**12, 2.5, 4.0
        r = np.linspace(rmin, rmax, m)
        x0, transient, steady = 0.5, 100, 2**10

        t1 = time()
        ss = parallel_logistic_map(r, x0, transient, steady)
        t2 = time()
        logistic_time = t2 - t1
        print('It took {} seconds to calculate the parallel_logistic_map graph.'.
              format(logistic_time))
        fig = plt.figure(3)
        plt.plot(r, ss, 'b.')
        plt.axis([rmin, rmax, 0, 1])
        plt.xlabel('r value')
        plt.ylabel('x value')
        plt.title('Iterations of the logistic map')
        plt.savefig("logistics map.png")

        x = np.zeros([m, steady])
        t1 = time()
        for j in range(r.shape[0]):
            tmp = logisticSteadyArray(x0, r[j], transient, steady)
            for i in range(steady):
                x[j,i] = tmp[i]
        t2 = time()
        logistic_time = t2 - t1
        print('It took {} seconds to calculate the serial_logistic_map graph.'.
              format(logistic_time))
    
    """
        a) Locate the `for` loops in the logistic map code. What quantity is iterated over in each loop?
        $ My Anwser >>>
            - Iterate over `iteration number`: range(transient) and range(steady)
            - Iterate over range(#r)

        b) For which loop are the computations in each iteration independent of one another? Briefly explain 
        the reasoning behind your response.
        $ My Anwser >>>
            - Each iteration in loop over #r is independent of one another, that why we can parallelize this loop.

        c) Implement a parallel version of the code to compute the bifurcation diagram of the logistic map.
        $ My Anwser >>>
            - As above

        d) Modify both the serial and parallel codes to print the time required to perform the computation. 
        Determine the time requried to compute 1000 iterations for 1000 parameter values, and report the 
        "acceleration" factor; i.e. the ratio of serial run time over parallel run time.
        $ My Anwser >>>
            - When r.size = 10**12, steady = 10**10
            - Acceleration = 2.6752564907073975/ 0.08854389190673828 = 30x
    """

    #Problem 4
    if Prob4:
        width = height = 2**16
        real_low = imag_low = -2
        imag_high = real_high = 2
        real_vals = np.linspace(real_low, real_high, width)
        imag_vals = np.linspace(imag_low, imag_high, height)

        t1 = time()
        mandel = parallel_mandelbrot(real_vals, imag_vals, 2.5, 256)
        t2 = time()
        mandel_time = t2 - t1

        fig = plt.figure(4, figsize=(90,90))
        plt.imshow(mandel, extent=(-2, 2, -2, 2))
        plt.colorbar()
        plt.savefig('parallel_mandelbrot.png')
        # plt.show()

        # boolean_mandel = mandel > 1
        # plt.imshow(boolean_mandel, extent=(-2, 2, -2, 2))
        # plt.savefig('boolean_mandel.png')
        # plt.show()

        print('It took {} seconds to calculate the parallel_mandelbrot graph.'.format(
            mandel_time))

    """
    a) What do the `for` loops in the serial code iterate over? For which loops are 
    the computations for each iteration independent of one another? Briefly explain 
    the reasoning behind your reply.
    $ My Anwser >>>
    - Iterate over nx and ny
    - both of them are independent cases, that's why both of them can be parallelized.

    b) Write a numba implementation that a lunches a 2D computational grid to parallelize 
    the appropriate loops. Run your parallel code and verify that it reproduces the 
    results of the serial version. 
    $ My Anwser >>>
    - As above
    
    Also run your code to answer the questions below:
    
    c) What is the finest resolution 2D grid can you run on your GPU? What error message 
    is generated by attempting finer grid resolution?
    $ My Anwser >>>    
    - Error Message: numba.cuda.cudadrv.driver.CudaAPIError: [2] Call to cuMemAlloc results in CUDA_ERROR_OUT_OF_MEMORY
    - width = height = 2**15

    d) What is the largest square block that you can run on your GPU? What error message 
    is generated if you request more threads in each block?
    $ My Anwser >>>    
    - Error Message: numba.cuda.cudadrv.driver.CudaAPIError: [1] Call to cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE
    - TPBX = TPBY = 2**7
    
    """



