from hw4_template import *

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
    kernel_dict = { "monte_carlo_kernel_sphere_vol": 
                        {   "kernel": monte_carlo_kernel_sphere_vol,
                            "true_rst": sphere_vol_true },
                    "monte_carlo_kernel_sphere_intertia": 
                        {   "kernel": monte_carlo_kernel_sphere_intertia,
                            "true_rst": sphere_intertia_true },
                    "monte_carlo_kernel_shell_vol": 
                        {   "kernel": monte_carlo_kernel_shell_vol,
                            "true_rst": shell_vol_true },
                    "monte_carlo_kernel_shell_intertia": 
                        {   "kernel": monte_carlo_kernel_shell_intertia,
                            "true_rst": shell_intertia_true }
                    }

    generate_error_plots(kernel_dict)
