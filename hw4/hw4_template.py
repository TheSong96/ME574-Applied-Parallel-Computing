from hw4_template import *
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit, cuda, float32, float64, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

RAD = 1
SH_N = 10


def plot_helper(u, name, levels=True):
    NX, NY = u.shape
    X = np.linspace(0.0, 1.0, NX, endpoint=True)
    Y = np.linspace(0.0, 1.0, NY, endpoint=True)
    XX, YY = np.meshgrid(X, Y)

    levels = np.linspace(-1, 1, 101, endpoint=True)
    plt.figure(name)
    if levels is not None:
        plt.contourf(XX, YY, u.T, levels=levels)
    else:
        plt.contourf(XX, YY, u.T)
    plt.axis([0, 1, 0, 1])
    plt.colorbar()
    plt.savefig(f"{name}.png")
    plt.show()


def plot_errors(iters_list, errors_list, fig_name="E(N)"):
    plt.figure(fig_name)
    plt.plot(iters_list, errors_list)
    plt.title(fig_name)
    plt.xscale('log')
    plt.savefig(fig_name)
    plt.show()


def generate_error_plots(kernel_dict):
    TPB = 2**6
    blocks = 24
    iters_list = [10**i for i in range(6)]
    for kernel_name in kernel_dict:
        kernel = kernel_dict[kernel_name]["kernel"]
        true_rst = kernel_dict[kernel_name]["true_rst"]
        errors_list = []
        for iters in iters_list:
            result = monte_carlo(TPB, blocks, iters, kernel)
            errors_list.append(np.abs(result - true_rst))
        plot_errors(iters_list, errors_list, kernel_name)


@cuda.reduce
def max_kernel(a, b):
    return max(a, b)


@cuda.reduce
def sum_kernel(a, b):
    return a + b


@cuda.jit
def heat_step(d_u, d_out, stencil, dt):
    """
        u: input device array
        out: output device array
        stencil: derivative stencil coefficients
        t: current time step
    """

    h = 1 / (d_u.shape[0] - 1)

    i, j = cuda.grid(2)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1]:
        return
    NX, NY = cuda.blockDim.x, cuda.blockDim.y
    t_i, t_j = cuda.threadIdx.x, cuda.threadIdx.y
    sh_i, sh_j = t_i + RAD, t_j + RAD

    sh_u = cuda.shared.array(shape=(SH_N, SH_N), dtype=float64)

    sh_u[sh_i, sh_j] = d_u[i, j]

    # Halo edge values assignment
    if t_i < RAD:
        sh_u[sh_i - RAD, sh_j] = d_u[i - RAD, j]
        sh_u[sh_i + NX, sh_j] = d_u[i + NX, j]

    if t_j < RAD:
        sh_u[sh_i, sh_j - RAD] = d_u[i, j - RAD]
        sh_u[sh_i, sh_j + NY] = d_u[i, j + NY]

    # Halo corner values assignment
    if t_i < RAD and t_j < RAD:
        # upper left
        sh_u[sh_i - RAD, sh_j - RAD] = d_u[i - RAD, j - RAD]
    # upper right
        sh_u[sh_i + NX, sh_j - RAD] = d_u[i + NX, j - RAD]
    # lower left
        sh_u[sh_i - RAD, sh_j + NY] = d_u[i - RAD, j + NY]
    # lower right
        sh_u[sh_i + NX, sh_j + NY] = d_u[i + NX, j + NY]

    cuda.syncthreads()

    if i > 0 and j > 0 and i < dims[0] - 1 and j < dims[1] - 1:
        # print("fk")
        d_out[i, j] = (sh_u[sh_i, sh_j-1] +
                       sh_u[sh_i, sh_j+1] +
                       sh_u[sh_i-1, sh_j] +
                       sh_u[sh_i+1, sh_j] -
                       4*sh_u[sh_i, sh_j]) / h / h * dt + sh_u[sh_i, sh_j]


def heat_update_parallel(u, stencil, dt):

    dims = u.shape
    d_u = cuda.to_device(u)
    d_stencil = cuda.to_device(stencil)
    d_out = cuda.device_array(d_u.shape, dtype=np.float64)
    dim = u.shape

    TPB = 8
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB]
    blockSize = [TPB, TPB]

    heat_step[gridSize, blockSize](d_u, d_out, d_stencil, dt)

    return d_out.copy_to_host()



# @cuda.jit(device=True)
def Si(x):
    if x == 0:
        return 1
    return math.sin(x) / x

@cuda.jit
def integrate_kernel(d_y, d_out, quad):
    '''
    y: input device array
    out: output device array
    quad: quadrature stencil coefficients
    '''
    i = cuda.grid(1)
    if i < d_out.size:
        d_out[i] = d_y[2*i] * quad[0] + d_y[2*i+1] * \
            quad[1] + d_y[2*i+2] * quad[2]


def integrate(y, quad):
    '''
    y: input array
    quad: quadrature stencil coefficients
    '''
    n = int((y.shape[0] - 1) / 2)
    d_y = cuda.to_device(y)
    d_out = cuda.device_array(n, dtype=np.float32)
    TPB = 32
    gridDim = (n+TPB-1)//TPB
    blockDim = TPB
    integrate_kernel[gridDim, blockDim](d_y, d_out,  quad)
    return np.sum(d_out.copy_to_host())


def RicharsonRecur(f, x, h, n):
    if n <= 2:
        return float(f(x + h) - f(x - h))/(2 * h)
    else:
        return float(2**(n - 2) * RicharsonRecur(f, x, h, n - 2) - 
                RicharsonRecur(f, x, 2 * h, n - 2)) / (2**(n-2) - 1)

@cuda.jit
def monte_carlo_kernel_sphere_intertia(rng_states, iters, d_out):
    '''
    rng_states: rng state array generated from xoroshiro random number generator
    iters: number of monte carlo sample points each thread will test
    out: output array
    '''
    i = cuda.grid(1)
    cnt = 0
    if i > d_out.shape[0]:
        return
    dd = 0.0
    for _ in range(iters):
        # such that x is in [-1, 1]
        x = (xoroshiro128p_uniform_float32(rng_states, i) - 0.5) * 2
        y = (xoroshiro128p_uniform_float32(rng_states, i) - 0.5) * 2
        z = (xoroshiro128p_uniform_float32(rng_states, i) - 0.5) * 2
        if x**2 + y**2 + z**2 <= 1:
            cnt += 1
            dd += (y**2 + z**2)
    # assuming m = 1
    d_out[i] = 4.0 / 3.0 * np.pi * dd / float(cnt)


@cuda.jit
def monte_carlo_kernel_sphere_vol(rng_states, iters, d_out):
    i = cuda.grid(1)
    cnt = 0
    for _ in range(iters):
        # such that x is in [-1, 1]
        x = (xoroshiro128p_uniform_float32(rng_states, i) - 0.5) * 2
        y = (xoroshiro128p_uniform_float32(rng_states, i) - 0.5) * 2
        z = (xoroshiro128p_uniform_float32(rng_states, i) - 0.5) * 2

        if x**2 + y**2 + z**2 <= 1:
            cnt += 1
    d_out[i] = 8 * cnt / iters


@cuda.jit
def monte_carlo_kernel_shell_intertia(rng_states, iters, d_out):
    """
        reference:  https://tutorial.math.lamar.edu/classes/calcIII/SurfaceIntegrals.aspx
                    http://www.twistedwg.com/2018/05/29/MC-integral.html
        assuming：surface density Rho = 1 kg/(m**2)
    """
    i = cuda.grid(1)
    inertia = 0.0     # d_inertia = Rho * dS * dd
    # (dd = square of distance from point on the shell to z-axis)
    cnt = 0.0
    Rho = 1
    s = 0.0  # projection of shell surface into x-y plane, which is a circle area
    for _ in range(iters):
        x = xoroshiro128p_uniform_float32(rng_states, i)
        y = xoroshiro128p_uniform_float32(rng_states, i)
        dd = x**2 + y**2
        if dd <= 1:
            dS = 1 / (1 - x**2 - y**2)**0.5
            dm = Rho * dS
            inertia += dm * dd
            cnt += 1.0
    s = cnt / iters
    d_out[i] = 8 * s * inertia / cnt


@cuda.jit
def monte_carlo_kernel_shell_vol(rng_states, iters, d_out):
    """
        reference:  https://tutorial.math.lamar.edu/classes/calcIII/SurfaceIntegrals.aspx
                    http://www.twistedwg.com/2018/05/29/MC-integral.html
        S2(3)   =   ∬dS=∬((∂z/∂x)^2+(∂z/∂y)^2+1)^(1/2)dxdy
                =   8 * ∬(1 - x**2 - y**2)^(1/2)dxdy
    """
    i = cuda.grid(1)
    S = 0.0  # shell surface area
    cnt = 0.0
    s = 0.0  # projection of shell surface into x-y plane, which is a circle area
    for _ in range(iters):
        x = xoroshiro128p_uniform_float32(rng_states, i)
        y = xoroshiro128p_uniform_float32(rng_states, i)
        if x**2 + y**2 <= 1:
            dS = 1 / (1 - x**2 - y**2)**0.5
            S += dS
            cnt += 1.0
    s = cnt / iters
    d_out[i] = 8 * s * S / cnt


def monte_carlo(TPB, blocks, iters, kernel, seed=1):
    '''
        threads: number of threads to use for the kernel
        blocks: number of blocks to use for the kernel
        iters: number of monte carlo sample points each thread will test 
        kernel: monte_carlo kernel to use
        seed: seed used when generating the random numbers (if the seed 
            is left at one the number generated will be the same each time)
    '''

    d_out = cuda.device_array(TPB * blocks, dtype=np.float32)
    rng_states = create_xoroshiro128p_states(TPB * blocks, seed)
    kernel[blocks, TPB](rng_states, iters, d_out)

    return np.mean(d_out.copy_to_host())


# @cuda.jit(device = True)
def chi(f, levelset):
    '''
    f: function value
    levelset: surface levelset
    '''
    return f <= levelset


# @cuda.jit
def grid_integrate_sphere_intertia(y, out, stencil):
    '''
    y: input device array
    out: output device array
    stencil: derivative stencil coefficients
    '''
    pass


# @cuda.jit
def grid_integrate_sphere_vol(y, out, stencil):
    pass


# @cuda.jit
def grid_integrate_shell_intertia(y, out, stencil):
    pass


# @cuda.jit
def grid_integrate_shell_vol(y, out, stencil):
    pass


def grid_integrate(kernel):
    '''
    kernel: grid integration kernel to use
    '''
    pass


if __name__ == "__main__":

    print("# ------------------------- Problem 1 ------------------------- #")
    dt = 0.00001
    n_iter = 10000
    NY, NX = 151, 151
    X = np.linspace(0.0, 1.0, NX, endpoint=True)
    Y = np.linspace(0.0, 1.0, NY, endpoint=True)
    h = 1.0/float(NX-1)

    u = np.zeros((NX, NY))
    # initial condition
    stencil = np.array([0.25, 0])
    XX, YY = np.meshgrid(X, Y)
    u = np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)
    u_max = np.max(u)

    plot_helper(u, "Problem1a_Init")

    max_iteration = 10000

    for i in range(max_iteration):
        u = heat_update_parallel(u, stencil, dt)
        if np.max(u) / u_max < np.exp(-2):
            print(f"find it at t2 = {i * dt} seconds")
            print(f"the largest time step is around = {dt}")
            plot_helper(u, "Problem_1b_result")
            break
        if np.isnan(np.max(u) / u_max):
            print("it's NAN!")
            plot_helper(u, "Problem_1c_result", levels=False)
            break

    # ----- problem 1c ----- #
    dt = 0.000013
    for i in range(max_iteration):
        u = heat_update_parallel(u, stencil, dt)
        if np.isnan(np.max(u) / u_max):
            print("When dt(0.000013) > dt_max(0.00001)")
            plot_helper(u, "Problem_1c_result", levels=False)
            break

    print("\n\n")
    print("# ------------------------- Problem 2 ------------------------- #")
    print("# --------------- using Simpson --------------- #")
    n = 6001
    a = 0
    b = 50
    h = (b-a)/float(n-1)
    arr = np.linspace(a, b, n, endpoint=True)
    for i in range(len(arr)):
        arr[i] = Si(arr[i])
    quar = h / 3 * np.array([1, 4, 1])
    rst = integrate(arr, quar)
    Si50 = 1.5516170724859358947279855  # (from mathematica)
    print(f'Si(50) true value : {Si50}')
    print(f'Si(50) using Simpson\'s rule : {rst}')
    print(f"Si(50) relative error rate = {np.abs((rst - Si50)/Si50)}")

    print("# --------------- using SimpsonWithRichar --------------- #")
    n = 6001
    a = 0
    b = 50
    h = (b-a)/float(n-1)
    arr = np.linspace(a, b, n, endpoint=True)
    arrRichar = np.zeros(2*n-1)
    for i in range(arr.shape[0] - 1):
        arrRichar[2*i] = Si(arr[i])
        arrRichar[2*i+1] = arrRichar[2*i] + h / \
            2.0 * RicharsonRecur(Si, arr[i], h, 2)
    arrRichar[-1] = Si(arr[-1])
    quar = h / 3 * np.array([1, 4, 1])

    rst = integrate(arrRichar, quar) / 2.0
    print(f'Si(50) using SimpsonWithRichar : {rst}')
    print(f"Si(50) relative error rate = {np.abs((rst - Si50)/Si50)}")

    print("\n\n")
    print("# ------------------------- Problem 3 ------------------------- #")
    TPB = 2**6
    blocks = 24
    iters = 10000
    R = 1
    Result_True_list = []

    print("# ------------------- sphere vol ------------------- #")
    kernel = monte_carlo_kernel_sphere_vol
    sphere_vol = monte_carlo(TPB, blocks, iters, kernel)
    sphere_vol_true = 4 / 3 * np.pi * R**2
    print(f"True Vol of Sphere = {sphere_vol_true}")
    print(f"MC Estimated Vol of Sphere = {sphere_vol}")

    print("# ------------------- sphere inertia ------------------- #")
    # assuming：volume density Rho = 1 kg/(m**3)
    Rho = 1
    m = Rho * 4 / 3 * np.pi * R**3
    sphere_intertia_true = 2 / 5 * m * R**2
    kernel = monte_carlo_kernel_sphere_intertia
    sphere_intertia = monte_carlo(TPB, blocks, iters, kernel)
    print(f"True Inertia of Sphere = {sphere_intertia_true}")
    print(f"MC Estimated Inertia of Sphere = {sphere_intertia}")

    """
        for shell vol and inertia, the most important part is to formula out
        the definite integral. then use monte carlo to estimate the definite 
        integral.
    """
    print("# ------------------- shell vol ------------------- #")
    shell_vol_true = 4 * np.pi * R**2
    kernel = monte_carlo_kernel_shell_vol
    shell_vol = monte_carlo(TPB, blocks, iters, kernel)
    print(f"True Vol of Shell = {shell_vol_true}")
    print(f"MC Estimated Vol of Shell = {shell_vol}")

    print("# ------------------- shell inertia ------------------- #")
    # assuming：surface density Rho = 1 kg/(m**2)
    Rho = 1
    m = Rho * 4 * np.pi * R**2
    shell_intertia_true = 2 / 3 * m * R**2
    kernel = monte_carlo_kernel_shell_intertia
    shell_inertia = monte_carlo(TPB, blocks, iters, kernel)
    print(f"True Inertia of Shell = {shell_intertia_true}")
    print(f"MC Estimated Inertia of Shell = {shell_inertia}")

    print("# ------------------- generate E(N) plots ------------------- #")
    kernel_dict = {"monte_carlo_kernel_sphere_vol":
                   {"kernel": monte_carlo_kernel_sphere_vol,
                    "true_rst": sphere_vol_true},
                   "monte_carlo_kernel_sphere_intertia":
                   {"kernel": monte_carlo_kernel_sphere_intertia,
                    "true_rst": sphere_intertia_true},
                   "monte_carlo_kernel_shell_vol":
                   {"kernel": monte_carlo_kernel_shell_vol,
                    "true_rst": shell_vol_true},
                   "monte_carlo_kernel_shell_intertia":
                   {"kernel": monte_carlo_kernel_shell_intertia,
                    "true_rst": shell_intertia_true}
                   }

    generate_error_plots(kernel_dict)
