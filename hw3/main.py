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

    """
        1c) according to the print result both the order of block and thread executions 
            are not predictable.
    """


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
    plt.plot(rst_serial, label =  'time of rad 2 = {}'.format(end - start))

    start = time()
    rst_serial = smooth_serial(w, 4)
    end = time()  

    plt.plot(rst_serial, label =  'time of rad 4 = {}'.format(end - start))
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
    print("time of parallel using cuda.event = {}".format(elapsed/1000))

    plt.plot(rst_parallel, label =  'time of rad 2 = {}'.format(elapsed))

    rst_parallel, elapsed = smooth_parallel(w, 4)
    plt.plot(rst_parallel, label =  'time of rad 4 = {}'.format(elapsed))
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
    print("time of smooth_parallel_sm using cuda.event = {}".format(elapsed/1000))

    plt.plot(rst_sm, label =  'time of rad 2 = {}'.format(elapsed))

    rst_sm, elapsed = smooth_parallel_sm(w, 4)
    plt.plot(rst_sm, label =  'time of rad 4 = {}'.format(elapsed))
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
    err_t100  = abs(rst[-1,0] - sinpi)
    plt.plot(t, rst[:,0], label = 'steps = {}, x = pi error = {:.2E}'.format(100,err_t100))

    # print("rst for Problem3_a with 101: \n", rst)

    t = np.linspace(0, np.pi, 11)
    rst = rk_solve(ode_f, Y0, t)
    err_t10 = abs(rst[-1,0]- sinpi)
    plt.plot(t, rst[:,0], label = 'steps = {}, x = pi error = {:.2E}'.format(10,err_t10))
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
    plt.savefig("find_eignvals")

    print("------------------------ Problem3 c ------------------------")

    c_list = np.linspace(0, 5, num=11, endpoint=True)
    eigenvals_list = []
    for c in c_list:
        eigenvals_list.append(find_eignvals(y[:, 1] - c*y[:, 0], w)) 

    eigenvals_list = np.array(eigenvals_list)
    print(eigenvals_list)

    """
        3b) the eignvalues found with the following resolution
            w = np.linspace(0, 10, num=1001, endpoint=True)
            is [1.29 2.37 3.4  4.42 5.44 6.45 7.45 8.46 9.46]
        3c) 
    """

    # d) OPTIONAL (should call function eigenvals_parallel)







    #Problem 4
    # (should call function jacobi_update_serial)

    NX,NY = 128,128

    xvals = np.linspace(-1.0, 1.0, NX)
    yvals = np.linspace(-1.0, 1.0, NY)

    u = np.zeros(shape=[NX,NY], dtype=np.float64)
    for i in range(NX):
        for j in range(NY):
            if xvals[i] == 0 and yvals[j] == 0:
                u[i][j] = 1
            if p4(xvals[i], yvals[j]) > 0:
                if xvals[i] > 0:
                    u[i,j] = 0.5
                if xvals[i] < 0: 
                    u[i,j] = -1

    for _ in range(100):
        u = jacobi_update_serial(xvals, yvals, u, p4)
        
    fig = plt.figure()
    axp = plt.axes(projection='3d')
    XX, YY = np.meshgrid(xvals, yvals)
    axp.contour3D(XX, YY, u, 100, cmap='viridis')
    plt.draw()
    plt.show()

    # pb4_plt(u.T,xvals,yvals)

    #Problem 5
    # (should call function jacobi_update_kernel)