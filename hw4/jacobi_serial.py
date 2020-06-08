#%%
import numpy as np
import matplotlib.pyplot as plt
from math import sin, sinh

NX, NY = 11,11
iters = NX*NX
PI = np.pi
method = 'Jacobi' #choices: 'Jacobi','Gauss-Seidel', 'SOR' 

def mean_update(u, method):
    '''
    update 2D array with non-boundary elements replaced by average of 4 nearest neighbors (on Cartesian grid)
    Args:
        u: 2D numpy array of floats
        method: string specifying 'Jacobi', 'Gauss-Seidel' or 'SOR'
    Returns:
        updated numpy array with same shape as u
    '''
    nx, ny = np.shape(u)

    if method == 'Jacobi':
        u_new = np.copy(u)
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                u_new[i,j] = (u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1])/4.
        return u_new

    if method == 'Gauss-Seidel':
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                u[i,j] = (u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1])/4.
        return u

    if method == 'SOR':
        h = 1./(nx-1)
        w = 2. * (1-2*PI*h)
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                u[i,j] =  u[i,j] + w*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]-4.*u[i,j])/4.
        return u

def main():
    #Compute exact solution
    exact = np.zeros(shape=[NX,NY], dtype=np.float32)
    for i in range(NX):
        for j in range(NY):
            exact[i,j]= sin(i*PI/(NX-1)) * sinh(j*PI/(NY-1))/sinh(PI)
    
    #serial iteration results
    u = np.zeros(shape=[NX,NY], dtype=np.float32)
    for i in range(NX):
        u[i,NX-1]= sin(i*PI/(NX-1))
    for k in range(1000):
        u = mean_update(u, method)
    
    error = np.max(np.abs(u-exact))
    print("%s, NX = %d, iters = %d => max error: %5.2e"  %(method, NX, iters, error))

    xvals = np.linspace(0., 1.0, NX)
    yvals = np.linspace(0., 1.0, NY)
    X,Y = np.meshgrid(xvals, yvals)
    levels = [0.025, 0.1, 0.25, 0.50, 0.75]
#     plt.contourf(X,Y,exact.T, levels = levels)
    plt.contourf(X,Y,u.T, levels = levels)
#     plt.contourf(X,Y,u.T, levels = levels,
#         colors = 'r', linewidths = 4)
    plt.axis([0,1,0,1])
    plt.show()

if __name__ == '__main__':
    main()

# %%
