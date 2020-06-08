import numpy as np
from numba import cuda
import numba
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

pi_string = "3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609"
print("len(pi_string) = ", len(pi_string))
pi_digits = [int(char) for char in pi_string]
v = 0.1 * np.array(pi_digits)[0:640]

global NSHARED


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
    # if cuda.threadIdx.x % 32 == 0:
    #     print(cuda.blockIdx.x, cuda.threadIdx.x)


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

    TPB = 96
    gridDim = (u.size + TPB - 1) // TPB
    blockDim = TPB

    ewprod_kernel[gridDim, blockDim](d_w, d_u, d_v)

    return d_w.copy_to_host()


def smooth_serial(v, rad):
    '''
    compute array of local averages (with radius rad)
    '''
    # rst = []
    # # print(v.shape)
    # for i in range(v.size):
    #     if i - rad < 0:
    #         local_array = v[0:i + rad + 1]
    #     elif i + rad + 1 > v.size:
    #         local_array = v[i - rad:v.size]
    #     else:
    #         local_array = v[i - rad:i + rad + 1]
    #     if local_array.shape[0] == 0:
    #         print("the {} th iteration is invalid".format(i))
    #     rst.append(local_array.mean())
    # rst = np.array(rst)

    rst = np.zeros_like(v)
    for i in range(rad, v.size - rad):
        stencil = v[i - rad:i + rad + 1]
        # tmp = 0
        # for k in stencil:
        #     tmp += k
        # rst[i] = tmp / stencil.size
        rst[i] = stencil.mean()
    return rst


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
        # if i - rad < 0:
        #     local_array = d_v[0:i + rad + 1]
        # elif i + rad + 1 > d_v.size:
        #     local_array = d_v[i - rad:d_v.size]
        # else:
        #     local_array = d_v[i - rad:i + rad + 1]

        if i - rad >= 0 and i + rad + 1 <= d_v.size:
            local_array = d_v[i - rad:i + rad + 1]
            d_w[i] = avg1(local_array)


def smooth_parallel(v, rad):
    '''
    compute array of local averages (with radius rad)
    Must call GPU kernel function
    '''
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(v.size, dtype=np.float64)

    TPB = 32
    gridDim = (v.size + TPB - 1) // TPB
    blockDim = TPB

    st = cuda.event()
    end = cuda.event()
    st.record()
    smooth_parallel_kernel[gridDim, blockDim](d_out, d_v, rad)
    end.record()
    end.synchronize()
    elapsed = cuda.event_elapsed_time(st, end)
    # print('elapsed time = {}'.format(elapsed))

    return d_out.copy_to_host(), elapsed


@cuda.jit
def smooth_sm_kernel(d_smooth, d_v, d_stencil, rad):
    n = d_v.shape[0]
    i = cuda.grid(1)
    global NSHARED
    # print("NSHARED = ", NSHARED)
    sh_f = cuda.shared.array(NSHARED, dtype=numba.float64)  # elements will be initialized with 0.0
    #thread index (and index for optional shared output array)
    tIdx = cuda.threadIdx.x
    bIdx = cuda.blockIdx.x

    shIdx = tIdx + rad

    if i >= n:
        return

    #Load regular cells
    sh_f[shIdx] = d_v[i]

    #Halo cells- Check that the entries to be loaded are within array bounds
    if tIdx < rad:
        if i >= rad:
            sh_f[shIdx - rad] = d_v[i - rad]
        if i + cuda.blockDim.x < n:
            sh_f[shIdx + cuda.blockDim.x] = d_v[i + cuda.blockDim.x]

    #make sure that shared array is fully loaded before any thread reads from it
    cuda.syncthreads()

    #write values only where the full stencil is "in bounds"
    if i >= rad and i < n - rad:
        stencil_dot = sh_f[shIdx] * d_stencil[rad]
        for d in range(1, rad + 1):
            stencil_dot += sh_f[shIdx - d] * d_stencil[rad - d] + \
                           sh_f[shIdx + d] * d_stencil[rad + d]
        d_smooth[i] = stencil_dot / d_stencil.size


def smooth_parallel_sm(v, rad):
    '''
    compute array of local averages (with radius rad)
    Must call GPU kernel function & utilize shared memory
    '''
    stencil = []
    for i in range(-rad, rad + 1):
        stencil.append(1)
    stencil = np.array(stencil)
    d_stencil = cuda.to_device(stencil)
    d_v = cuda.to_device(v)
    d_smooth = cuda.device_array(v.size, dtype=np.float64)

    TPB = 16
    gridDim = (v.size + TPB - 1) // TPB
    blockDim = TPB
    global NSHARED
    NSHARED = 2 * rad + TPB

    st = cuda.event()
    end = cuda.event()
    st.record()
    smooth_sm_kernel[gridDim, blockDim](d_smooth, d_v, d_stencil, rad)
    end.record()
    end.synchronize()
    elapsed = cuda.event_elapsed_time(st, end)

    return d_smooth.copy_to_host(), elapsed


def ode_f(t, y):
    """
    Compute right hand side of dy = f(t, y)

    Args:
        y: np.array 

    Returns:
        d_y: d_y at y
    """
    w = 1
    A = np.array([[0, 1], [-w**2, 0]])
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
    y_list = [y]
    for i in range(t.shape[0] - 1):
        y = rk4_step(f, y, t[i], h)
        y_list.append(y)

    return np.array(y_list)


@cuda.jit(device=True)
def ode_f_parallel(t, y0, y1, w):

    return y1, -y0 * w * w


@cuda.jit
def rk4_step_parallel(Y_all, y, t, h, w):
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
    i = cuda.grid(1)
    if i < y.shape[0]:
        for k in range(t.shape[0] - 1):
            f1 = f(t[k], y[i][0], y[i][1], w[i])
            f2 = f(t[k] + 0.5 * h, y[i][0] + 0.5 * h * f1[0], y[i][1] + 0.5 * h * f1[1], w[i])
            f3 = f(t[k] + 0.5 * h, y[i][0] + 0.5 * h * f2[0], y[i][1] + 0.5 * h * f2[1], w[i])
            f4 = f(t[k] + h, y[i][0] + h * f3[0], y[i][1] + h * f3[1], w[i])
            for j in range(y.shape[1]):
                y[i][j] += h / 6 * (f1[j] + 2 * f2[j] + 2 * f3[j] + f4[j])
                Y_all[i][k + 1][j] = y[i][j]


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
    y_dev = cuda.to_device(y)  #cuda.to_device() will transform a single value to an array
    w_dev = cuda.to_device(w)
    t_dev = cuda.to_device(t)

    Y_all = np.zeros((w_dev.size, t_dev.size, y_dev.shape[1]), np.float64)
    Y_all[:, 0, :] = y
    Y_all = cuda.to_device(Y_all)

    TPB = 32
    gridDim = (y.size + TPB - 1) // TPB
    blockDim = TPB
    rk4_step_parallel[gridDim, blockDim](Y_all, y_dev, t_dev, h, w_dev)

    return y_dev.copy_to_host()


def find_eignvals(diff, w):
    """
        find sign changes in diff
    """
    eignvals = []
    for i in range(diff.shape[0] - 1):
        if diff[i] * diff[i + 1] < 0:
            eignvals.append(w[i])
    return np.array(eignvals)


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


def p4(x, y):
    return np.power(x, 4) + np.power(y, 4) - 1


def jacobi_update_serial(x, y, u, f):
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
    v = np.copy(u)
    for i in range(len(x)):
        for j in range(len(y)):
            if f(x[i], y[j]) <= 0:
                if x[i] == 0 and y[j] == 0:
                    v[i, j] = 1
                else:
                    v[i, j] = 1 / 4 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
    return v


@cuda.jit(device=True)
def p4_dev(x, y):

    return x**4 + y**4 - 1


@cuda.jit
def jacobi_update_kernel(d_u, u_out, x, y):
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

    i, j = cuda.grid(2)

    if i < x.shape[0] and j < y.shape[0]:
        if p4_dev(x[i], y[j]) <= 0:
            if x[i] == 0 and y[j] == 0:
                u_out[i, j] = 1
            else:
                u_out[i, j] = 1 / 4 * (d_u[i - 1, j] + d_u[i + 1, j] + d_u[i, j - 1] +
                                       d_u[i, j + 1])
        else:
            u_out[i, j] = d_u[i, j]


def jacobi_update_parallel(x, y, u):

    TPBY = 16
    TPBX = 16
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    u = cuda.to_device(u)
    u_out = cuda.device_array(u.shape, dtype=np.float64)
    gridDims = ((len(x) + TPBX - 1) // TPBX, (len(y) + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)
    jacobi_update_kernel[gridDims, blockDims](u, u_out, d_x, d_y)

    return u_out.copy_to_host()


if __name__ == "__main__":

    # Problem 1
    print("------------------------ Problem 1 ------------------------")
    #(should call function ewprod)
    # v = np.zeros(2**27)
    start = time()
    rst = ewprod_for(v, 1.0 - v)
    end = time()
    print("time of for loop = ", end - start)

    start = time()
    rst = ewprod(v, 1.0 - v)
    end = time()
    print("time of numba parallel = ", end - start)

    start = time()
    rst = v * (1.0 - v)
    end = time()
    print("time of built-in numpy = ", end - start)

    """
        1c) according to the print result both the order of block and thread executions
            are not predictable.
    """

    print("1c) according to the print result both the order of block and \
        thread executions are not predictable.")

    #Problem 2
    print("------------------------ Problem 2 ------------------------")
    # a) (should call function smooth_serial)
    rad = 1

    w = np.outer(v, 1 - v).flatten()
    # w = np.arange(9)
    plt.figure(0)

    start = time()
    rst_serial = smooth_serial(w, 2)
    end = time()
    print("time of smooth = ", end - start)
    plt.plot(rst_serial, label='time of rad 2 = {}'.format(end - start))

    start = time()
    rst_serial = smooth_serial(w, 4)
    end = time()

    plt.plot(rst_serial, label='time of rad 4 = {}'.format(end - start))
    plt.legend(loc='upper right')
    plt.title('Smooth Serial')
    plt.savefig("Problem2_a.png")
    plt.show()

    # b) (should call function smooth_parallel)
    plt.figure(1)

    start = time()
    rst_parallel, elapsed = smooth_parallel(w, 2)
    end = time()
    print("time of parallel = ", end - start)
    print("time of parallel using cuda.event = {}".format(elapsed / 1000))

    plt.plot(rst_parallel, label='time of rad 2 = {}'.format(elapsed))

    rst_parallel, elapsed = smooth_parallel(w, 4)
    plt.plot(rst_parallel, label='time of rad 4 = {}'.format(elapsed))
    plt.legend(loc='upper right')
    plt.title('Smooth parallel')
    plt.savefig("Problem2_b.png")
    plt.show()

    # c) (should call function smooth_parallel_sm)
    plt.figure(2)

    start = time()
    rst_sm, elapsed = smooth_parallel_sm(w, 2)
    end = time()
    print("time of smooth_parallel_sm = ", end - start)
    print("time of smooth_parallel_sm using cuda.event = {}".format(elapsed / 1000))

    plt.plot(rst_sm, label='time of rad 2 = {}'.format(elapsed))

    rst_sm, elapsed = smooth_parallel_sm(w, 4)
    plt.plot(rst_sm, label='time of rad 4 = {}'.format(elapsed))
    plt.legend(loc='upper right')
    plt.title('Smooth Parallel Shared Memory')
    plt.savefig("Problem2_c.png")
    plt.show()

    # assert (rst_serial==rst_parallel).all()
    # assert (rst_serial==rst_sm).all()
    # assert (rst_parallel==rst_sm).all()

    #Problem 3
    print("------------------------ Problem 3 ------------------------")
    # a) (should call function rk4_solve)
    Y0 = np.array([0.0, 1.0])
    sinpi = 0

    t = np.linspace(0, np.pi, 101)
    rst = rk_solve(ode_f, Y0, t)

    plt.figure(3)
    err_t100 = abs(rst[-1, 0] - sinpi)
    plt.plot(t, rst[:, 0], label='steps = {}, x = pi error = {:.2E}'.format(100, err_t100))

    # print("rst for Problem3_a with 101: \n", rst)

    t = np.linspace(0, np.pi, 11)
    rst = rk_solve(ode_f, Y0, t)
    err_t10 = abs(rst[-1, 0] - sinpi)
    plt.plot(t, rst[:, 0], label='steps = {}, x = pi error = {:.2E}'.format(10, err_t10))
    plt.title('Problem3_a')
    plt.legend(loc="lower center")
    plt.savefig("Problem3_a.png")
    plt.show()
    # print("rst for Problem3_a with 11: \n", rst)

    print("------------------------ Problem3 b ------------------------")
    # b/c) (should call function rk4_solve_parallel)
    w = np.linspace(0, 10, num=1001, endpoint=True)
    Y0 = np.array([Y0] * w.size)
    t = np.linspace(0, np.pi, 101)
    y = rk4_solve_parallel(Y0, t, w)
    diff = y[:, 1] - y[:, 0]
    eigenvals = find_eignvals(diff, w)
    print("eigenvals for Problem3_b: \n", eigenvals)

    plt.figure(4)
    plt.plot(w, diff)
    plt.title("find eigenvals for c=1")
    plt.savefig("find_eignvals")
    plt.show()

    print("------------------------ Problem3 c ------------------------")

    c_list = np.linspace(0, 5, num=11, endpoint=True)
    eigenvals_list = []
    diff_list = np.zeros((len(c_list), diff.shape[0]))
    for i, c in enumerate(c_list):
        diff_list[i] = y[:, 1] - c * y[:, 0]
        eigenvals_list.append(find_eignvals(y[:, 1] - c * y[:, 0], w))

    eigenvals_list = np.array(eigenvals_list)
    print("eigenvals_list for different c =\n", eigenvals_list)
    plt.figure(5)
    plt.plot(w, diff_list.T)
    plt.title("zero point of these curves are eignvalues for different c = [0, 5]")
    plt.savefig("find_eignvals")
    plt.show()

    """
        3b) the eignvalues found with the following resolution
            w = np.linspace(0, 10, num=1001, endpoint=True)
            is [1.29 2.37 3.4  4.42 5.44 6.45 7.45 8.46 9.46]
        3c)
    """

    # d) OPTIONAL (should call function eigenvals_parallel)

    print("------------------------ Problem 4  ------------------------")

    #Problem 4
    # (should call function jacobi_update_serial)

    NX, NY = 128, 128

    xvals = np.linspace(-1.0, 1.0, NX)
    yvals = np.linspace(-1.0, 1.0, NY)

    start = time()
    u = np.zeros(shape=[NX, NY], dtype=np.float64)
    for i in range(NX):
        for j in range(NY):
            if xvals[i] == 0 and yvals[j] == 0:
                u[i][j] = 1
            if p4(xvals[i], yvals[j]) > 0:
                if xvals[i] > 0:
                    u[i, j] = 0.5
                if xvals[i] < 0:
                    u[i, j] = -1

    for _ in range(100):
        u = jacobi_update_serial(xvals, yvals, u, p4)
    elapsed4 = time() - start
    print('Execution time take for Problem 4:' + str(elapsed4) + 'seconds')

    plt.figure(6)
    axp = plt.axes(projection='3d')
    XX, YY = np.meshgrid(xvals, yvals)
    axp.contour3D(XX, YY, u, 100, cmap='viridis')
    plt.title("Problem4")
    plt.savefig("Problem4.png")
    plt.draw()
    plt.show()

    print("------------------------ Problem 5  ------------------------")

    #Problem 5
    # (should call function jacobi_update_kernel)
    start = time()

    u = np.zeros(shape=[NX, NY], dtype=np.float64)
    for i in range(NX):
        for j in range(NY):
            if xvals[i] == 0 and yvals[j] == 0:
                u[i][j] = 1
            if p4(xvals[i], yvals[j]) > 0:
                if xvals[i] > 0:
                    u[i, j] = 0.5
                if xvals[i] < 0:
                    u[i, j] = -1

    for _ in range(100):
        u = jacobi_update_parallel(xvals, yvals, u)

    elapsed5 = time() - start
    print('Execution time take for Problem 5:' + str(elapsed5) + 'seconds')
    plt.figure(7)
    axp = plt.axes(projection='3d')
    XX, YY = np.meshgrid(xvals, yvals)
    axp.contour3D(XX, YY, u, 100, cmap='viridis')
    plt.title("Problem5")
    plt.savefig("Problem5.png")
    plt.draw()
    plt.show()

    print("acceleration factor = {}".format(elapsed4/elapsed5))

    """
         acceleration factor = 7.470 / 0.525 = 14.23
    """
