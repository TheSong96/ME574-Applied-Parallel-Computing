import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from time import time

pi_string = "3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609"
print("len(pi_string) = ", len(pi_string))
pi_digits = [int(char) for char in pi_string]
v = 0.1 * np.array(pi_digits)[0:640]


def ewprod_for(u, v):
    rst = np.zeros(u.size)
    for i in range(len(u)):
        rst[i] = u[i] * v[i]
    return rst


@cuda.jit(device=True)
def f(x, y):
    """
    Compute element-wise mutiplication
    """
    return x * y


@cuda.jit
def ewprod_kernel(d_w, d_u, d_v):

    i = cuda.grid(1)
    if i < d_u.size:
        d_w[i] = f(d_u[i], d_v[i])


def ewprod(u, v):
    '''
    Compute the element-wise product of arrays
    Must call GPU kernel function
    Args:
        u,v: 1D numpy arrays

    Returns:
        w: 1D numpy array containing product of corresponding entries in input arrays u and v
    '''
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_w = cuda.device_array(u.size, dtype=np.float64)

    TPB = 32
    gridDim = (u.size + TPB - 1) // TPB
    blockDim = TPB

    ewprod_kernel[gridDim, blockDim](d_w, d_u, d_v)

    return d_w.copy_to_host()


def smooth_serial(v, rad):
    '''
    compute array of local averages (with radius rad)
    '''
    rst = []
    # print(v.shape)
    for i in range(v.size):
        if i - rad < 0:
            local_array = v[0:i + rad + 1]
        elif i + rad + 1 > v.size:
            local_array = v[i - rad:v.size]
        else:
            local_array = v[i - rad:i + rad + 1]
        if local_array.shape[0] == 0:
            print("the {} th iteration is invalid".format(i))
        rst.append(local_array.mean())

    return np.array(rst)


@cuda.jit(device=True)
def avg1(x):
    y = 0
    for i in range(x.size):
        y += x[i]
    return y / x.size


@cuda.jit
def smooth_parallel_kernel(d_w, d_v, rad):

    i = cuda.grid(1)
    if i < d_w.size:
        # #---- method 1 ----#
        # d_w[i] = 0
        # if i - rad < 0:
        #     for k in range(0, i + rad + 1)
        #         d_w[i] += d_v[k]
        #     d_w /= i + rad + 1
        # elif i + rad + 1 > d_v.size:
        #     for k in range(i - rad, d_v.size)
        #         d_w[i] += d_v[k]
        #     local_array = d_v[i - rad:v.size]
        # else:
        #     local_array = d_v[i - rad:i + rad + 1]
        # d_w[i] = avg1(local_array)
        # local_array = d_v[0:2]
        # avg1(local_array)
        # d_w[i] = local_array.mean()

        # #---- method 2 ----#
        if i - rad < 0:
            local_array = d_v[0:i + rad + 1]
        elif i + rad + 1 > d_v.size:
            local_array = d_v[i - rad:d_v.size]
        else:
            local_array = d_v[i - rad:i + rad + 1]

        # y = 0
        # for k in range(local_array.size):
        #     y += local_array[k]
        # y /= local_array.size
        # d_w[i] = y

        d_w[i] = avg1(local_array)


def smooth_parallel(v, rad):
    '''
    compute array of local averages (with radius rad)
    Must call GPU kernel function
    '''
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(v.size, dtype=np.float64)
    # d_rad = cuda.to_device(rad)

    # print(type(d_rad))

    TPB = 32
    gridDim = (v.size + TPB - 1) // TPB
    blockDim = TPB
    smooth_parallel_kernel[gridDim, blockDim](d_out, d_v, rad)

    return d_out.copy_to_host()


def smooth_parallel_sm(v, rad):
    '''
    compute array of local averages (with radius rad)
    Must call GPU kernel function & utilize shared memory
    '''

    return d_out.copy_to_host()


def ode_f(t, y):
    """
    Compute right hand side of dy = f(t, y)

    Args:
        y: np.array 

    Returns:
        d_y: d_y at y
    """
    A = np.array([[0, 1], [-1, 0]])
    d_y = np.dot(A, y)

    return d_y


def rk4_step(f, y, t0, h):
    """
    compute next value for 4th order Runge Kutta ODE solver

    Args:
        f: right-hand side function that gives rate of change of y
        y: float value of dependent variable
        t0: float initial value of independent variable
        h: float time step

    Returns:
        y_new: float estimated value of y(t0+h)
    """
    f1 = f(t0, y)
    f2 = f(t0 + h / 2, y + h / 2 * f1)
    f3 = f(t0 + h / 2, y + h / 2 * f2)
    f4 = f(t0 + h, y + h * f3)
    y_new = y + h / 6 * (f1 + 2 * f2 + 2 * f3 + f4)

    return y_new


def rk_solve(f, y0, t):
    """
    Runge-Kutta solver for systems of 1st order ODEs
    (should call function rk4_step)

    Args:
        f: name of right-hand side function that gives rate of change of y
        y0: numpy array of initial float values of dependent variable
        t: numpy array of float values of independent variable

    Returns:
        y: 2D numpy array of float values of dependent variable
    """
    h = t[1] - t[0]
    y = y0
    y_list = []
    for t0 in t:
        y = rk4_step(f, y, t0, h)
        y_list.append(y)

    return np.array(y_list)


@cuda.jit(device=True)
def ode_f_parallel(t, y0, y1, w):
    # print(y1)
    # if w!=0:
    #     print(- y0 * w * w)
    return y1, - y0 * w * w


@cuda.jit
def rk4_step_parallel(y, t, h, w):
    """
    compute next value for 4th order Runge Kutta ODE solver

    Args:
        f: right-hand side function that gives rate of change of y
        y: float value of dependent variable
        t: float initial value of independent variable
        h: float time step
        w: float of omega value
    """
    f = ode_f_parallel
    # print(y.shape[0])
    i = cuda.grid(1)
    if i < y.shape[0]:
        for k in range(t.shape[0]):
            # print("k = ", k)
            # if w[i] > 9:
            #     print("w[i] = ", w[i])
            f1 = f(t[k], y[i][0], y[i][1], w[i])
            f2 = f(t[k] + h / 2, y[i][0] + h / 2 * f1[0],
                   y[i][1] + h / 2 * f1[1], w[i])
            f3 = f(t[k] + h / 2, y[i][0] + h / 2 * f2[0],
                   y[i][1] + h / 2 * f2[1], w[i])
            f4 = f(t[k] + h, y[i][0] + h * f3[0], y[i][1] + h * f3[1], w[i])
            # for j in range(y.shape[1]):
            # print(f1[j])
            # y[i][j] += h / 6 * (f1[j] + 2 * f2[j] + 2 * f3[j] + f4[j])
            # if f1[0] > 0.5:
            #     print("f1[0], f1[1] = ", f1[0], f1[1])
            y[i][0] += h / 6 * (f1[0] + 2 * f2[0] + 2 * f3[0] + f4[0])
            y[i][1] += h / 6 * (f1[1] + 2 * f2[1] + 2 * f3[1] + f4[1])


# @cuda.jit
def rk4_solve_parallel(y, t, w):
    """
    Runge-Kutta solver for systems of 1st order ODEs
    (should call function rk4_step_parallel)

    Args:
        f: name of right-hand side function that gives rate of change of y
        y: numpy array of dependent variable output (fist entry should be initial conditions)
        t: numpy array of float values of independent variable
        w: numpy array of omega values
    """
    h = t[1] - t[0]
    # h = cuda.to_device(t[1] - t[0])
    # print(h[0])
    """
        if it's just a single float value intead of an array,
        why it's not working in kernel? eg: cannot add.
    """
    y_dev = cuda.to_device(y)
    w_dev = cuda.to_device(w)
    t_dev = cuda.to_device(t)
    # print(y_dev.__cuda_array_interface__)
    TPB = 32
    gridDim = (y.size + TPB - 1) // TPB
    blockDim = TPB
    rk4_step_parallel[gridDim, blockDim](y_dev, t_dev, h, w_dev)

    return y_dev.copy_to_host()


#OPTIONAL#
def eigenvals_parallel(NC, NW, NT, h):
    """
    Determines eigenvalues of y''+w^2*y = 0, y'(pi) -c*y(pi) = 0 for w : [0,10] and c : [0,5]
    y(0) = 0, y'(0) = 1
    (should call a 2D gpu kernel)

    Args:
        NC: Number of samples in the range [0,5] to take for c
        NW: Number of samples in the range [0,10] to take for w
        NT: Number of timesteps to run each instance of the GPU-based rk4 solver
        h: step size for rk4 solver

    Returns:
        eigs: 2D numpy array of shape NCxNW containing eigenvalues as a function of c and w
    """

    #return eigs
    pass


##########


def jacobi_update_serial(u, f):
    """
    Performs jacobi iteration update for 2D Poisson equation
    Boundary conditions: u(0,0)=1; outside boundary defined by f, u(x<0)=-1, u(x>0)=0.5

    Args:
        x: vector specifying x coordinates
        y: vector specifying y coordinates
        u: 2D input array for update (u.shape = (len(x),len(y)))
        f: function specifying boundary of region to evaluation Poission equation

    Precondtion:
        The cordinate system defined by vectors x and y contains the point (0,0) and spans 
        the entire boundary defined by f

    Returns:
        v: updated version of u
    """
    return v


@cuda.jit
def jacobi_update_kernel(d_u, d_x, d_y):
    """
    Performs jacobi iteration update for 2D Poisson equation
    Boundary conditions: u(0,0)=1; outside boundary defined by f, u(x<0)=-1, u(x>0)=0.5
    where f = x^4 + y^4 - 1

    Args:
        d_x: vector specifying x coordinates
        d_y: vector specifying y coordinates
        d_u: 2D input array for update (u.shape = (len(x),len(y)))

    Precondtion:
        The cordinate system defined by vectors x and y contains the point (0,0) and spans 
        the entire boundary defined by f
    """
    pass
