from hw3 import *

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

    #Problem 2
    print("------------------------ Problem 2 ------------------------")
    # a) (should call function smooth_serial)
    rad = 2
    v = np.zeros(2**5) 
    w = np.outer(v, 1 - v).flatten()
    # print("w.shape = ", w.shape)

    start = time()
    rst = smooth_serial(w, rad)
    # rst[:] = 1
    end = time()    
    print("time of smooth = ", end - start)

    # print(rst)

    # b) (should call function smooth_parallel)
    start = time()
    rst_parallel = smooth_parallel(w, rad)
    assert (rst_parallel==rst).all()
    end = time()
    print("time of smooth_parallel = ", end - start)

    # c) (should call function smooth_parallel_sm)
    # start = time()
    # rst = smooth_parallel_sm(v, rad)
    # end = time()
    # print("time of for smooth_parallel_sm = ", end - start)

    #Problem 3
    print("------------------------ Problem 3 ------------------------")    
    # a) (should call function rk4_solve)
    Y0 = np.array([0, 1])

    t = np.linspace(0, np.pi, 100)
    rst = rk_solve(ode_f, Y0, t)
    plt.plot(t, rst[:, 0])

    t = np.linspace(0, np.pi, 10)
    rst = rk_solve(ode_f, Y0, t)

    plt.plot(t, rst[:, 0])
    plt.savefig("Problem3_a")

    print(rst)
    # b/c) (should call function rk4_solve_parallel)
    w = np.linspace(0, 10, 11)
    print("w = ", w)
    # w = np.array(1)
    Y0 = np.array([Y0] * w.size)
    # print(Y0.shape)
    rst = rk4_solve_parallel(Y0, t, w)
    print(rst)

    # d) OPTIONAL (should call function eigenvals_parallel)

    #Problem 4
    # (should call function jacobi_update_serial)

    #Problem 5
    # (should call function jacobi_update_kernel)