import numpy as np
import math
import matplotlib.pyplot as plt
from time import time

def dn_cos(n):
    if n == 0:
        return np.cos(0)
    else:
        return -dn_sin(n - 1)


def dn_sin(n):
    '''
    Compute the n^th derivative of sin(x) at x=0
    input:
        n - int: the order of the derivative to compute
    output:
        float nth derivative of sin(0)
    '''
    if n == 0:
        return np.sin(0)
    else:
        return dn_cos(n - 1)

def taylor_sin(x, n):
    '''
    Evaluate the Taylor series of sin(x) about x=0 neglecting terms of order x^n

    input:
        x - float: argument of sin
        n - int: number of terms of the taylor series to use in approximation
    output:
        float value computed using the taylor series truncated at the nth term
    '''
    assert n >= 2

    def series(x, m):
        if x == 0:
            return 0
        if m == 0:
            return x
        
        return (-1)**m * x**(2*m + 1) / np.math.factorial(2*m + 1) + series(x, m-1)

    m = int(n/2) - 1

    return series(x, m)

def measure_diff(ary1, ary2):
    '''
    Compute a scalar measure of difference between 2 arrays

    input:
        ary1 - numpy array of float values
        ary2 - numpy array of float values
    output:
        a float scalar quantifying difference between the arrays
    '''
    ary1 = np.array(ary1)
    ary2 = np.array(ary2)
    return np.linalg.norm(ary1-ary2)


def escape(cx, cy, dist, itrs, x0=0, y0=0):
    '''
    Compute the number of iterations of the logistic map, 
    f(x+j*y)=(x+j*y)**2 + cx +j*cy with initial values x0 and y0 
    with default values of 0, to escape from a cirle centered at the origin.

    inputs:
        cx - float: the real component of the parameter value
        cy - float: the imag component of the parameter value
        dist: radius of the circle
        itrs: int max number of iterations to compute
        x0: initial value of x; default value 0
        y0: initial value of y; default value 0
    returns:
        an int scalar interation count
    '''
    c = np.complex64( cx + cy * 1j  )            
    z = np.complex64(x0 + y0 * 1j)
    for i in range(itrs):
        z = z**2 + c
        if(np.abs(z) > dist):
            return i
    return itrs

def mandelbrot(cx, cy, dist, itrs):
    '''
    Compute escape iteration counts for an array of parameter values

    input:
        cx - array: 1d array of real part of parameter
        cy - array: 1d array of imaginary part of parameter
        dist - float: radius of circle for escape
        itrs - int: maximum number of iterations to compute
    output:
        a 2d array of iteration count for each parameter value (indexed pair of values cx, cy)
    '''


    mandelbrot_graph = np.ones((height,width), dtype=np.float32)
    
    for x in range(len(cx)):
        
        for y in range(len(cy)):
            
            mandelbrot_graph[y,x] = escape(cx[x], cy[y], dist, itrs, x0=0, y0=0)
    
    return mandelbrot_graph



if __name__ == '__main__':

    #Problem 2/3
    x = np.arange(-np.pi,np.pi,0.1)   # start,stop,step
    y = np.sin(x)
    plt.figure(figsize=(15,10))
    plt.plot(x,y, label="sin(x)")

    for i in range(2, 16, 2):
        y = []
        for dx in x:
            y.append(taylor_sin(dx, i))
        y = np.array(y)
        plt.plot(x, y, label=str(i) + "th")
    plt.legend(loc='best')
    plt.show()

    #Problem 4
    error_limit = 1e-2

    n_list = []
    for i in  range(2,16,2):
        n_list.append(i)

    x = np.linspace(0, np.pi/4, 50)
    y_true = np.sin(x)
            
    def find_n():
        n = 2
        while(True):
            y = []
            for dx in x:
                y.append(taylor_sin(dx, n))
            y = np.array(y)
            diff = measure_diff(y, y_true)
            if diff < error_limit:
                break
            n += 2
        return n
    n = find_n()
    print("n = ", n)

    #Problem 5
    width = height = 512
    real_low = imag_low = -2
    imag_high = real_high = 2
    real_vals = np.linspace(real_low, real_high, width)
    imag_vals = np.linspace(imag_low, imag_high, height)

    t1 = time()
    mandel = mandelbrot(real_vals, imag_vals, 2.5, 256)
    t2 = time()
    mandel_time = t2 - t1
    
    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.colorbar()
    plt.savefig('mandelbrot.png')
    plt.show()
    t2 = time()

    boolean_mandel = mandel > 1
    plt.imshow(boolean_mandel, extent=(-2, 2, -2, 2))
    plt.savefig('boolean_mandel.png')
    plt.show()

    
    dump_time = t2 - t1
    
    print('It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time))
    print('It took {} seconds to dump the image.'.format(dump_time))
