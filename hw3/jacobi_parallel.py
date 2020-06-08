# #%%
import numpy as np
import matplotlib.pyplot as plt
from math import sin, sinh
from numba import jit, cuda, float32, int32

NX, NY = 21, 21
iters = NX*NX//2
PI = np.pi
STENCIL_POINTS = 9 #specify 5 or 9 for points in stencil
TPB = 8
RAD = 1
SH_N = 10

#kernel with 2D shared memory array including halo
@cuda.jit
def updateKernel(d_v, d_u, edge, corner):
    i,j = cuda.grid(2)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1]:
        return
    NX, NY = cuda.blockDim.x, cuda.blockDim.y
    t_i, t_j = cuda.threadIdx.x, cuda.threadIdx.y
    sh_i, sh_j = t_i + RAD, t_j + RAD

    sh_u = cuda.shared.array(shape = (SH_N,SH_N), dtype = float32)

    #Load regular values
    sh_u[sh_i, sh_j] = d_u[i, j]
    
    #Halo edge values
    if t_i<RAD:
        sh_u[sh_i - RAD, sh_j] = d_u[i-RAD, j]
        sh_u[sh_i + NX , sh_j] = d_u[i+NX , j]

    if t_j<RAD:
        sh_u[sh_i, sh_j - RAD] = d_u[i, j - RAD]
        sh_u[sh_i, sh_j + NY ] = d_u[i, j + NY ]

    #Halo corner values
    if t_i<RAD and t_j<RAD:
        #upper left
        sh_u[sh_i - RAD, sh_j - RAD] = d_u[i-RAD, j - RAD]
        sh_u[sh_i - RAD, sh_j - RAD] = d_u[i-RAD, j - RAD]
        #upper right
        sh_u[sh_i + NX, sh_j - RAD] = d_u[i + NX, j - RAD]
        sh_u[sh_i + NX, sh_j - RAD] = d_u[i + NX, j - RAD]
        #lower left
        sh_u[sh_i - RAD, sh_j + NY] = d_u[i-RAD, j + NY]
        sh_u[sh_i - RAD, sh_j + NY] = d_u[i-RAD, j + NY]
        #lower right
        sh_u[sh_i + NX, sh_j + NX] = d_u[i + NX, j + NY]
        sh_u[sh_i + NX, sh_j + NY] = d_u[i + NX, j + NY]

    cuda.syncthreads()

    if i>0 and j>0 and i<dims[0]-1 and j<dims[1]-1:
        d_v[i, j] = \
            sh_u[sh_i-1, sh_j -1]*corner + \
            sh_u[sh_i, sh_j -1]*edge + \
            sh_u[sh_i+1, sh_j -1]*corner + \
            sh_u[sh_i-1, sh_j]*edge + \
            sh_u[sh_i, sh_j]*0. + \
            sh_u[sh_i+1, sh_j]*edge + \
            sh_u[sh_i-1, sh_j +1]*corner + \
            sh_u[sh_i, sh_j + 1] * edge + \
            sh_u[sh_i+1, sh_j +1]*corner
            # edge * (sh_u[sh_i-1, sh_j] + sh_u[sh_i+1, sh_j] + \
            #     sh_u[sh_i, sh_j-1] + sh_u[sh_i, sh_j+1]) + \
            # corner * (sh_u[sh_i-1, sh_j -1] + sh_u[sh_i+1, sh_j +1] + \
            #     sh_u[sh_i-1, sh_j +1] + sh_u[sh_i+1, sh_j-1])

def update(u, iter_count, edge, corner):
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(u)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB]
    blockSize = [TPB, TPB]

    for k in range(iter_count):
        updateKernel[gridSize, blockSize](d_v, d_u, edge, corner)
        updateKernel[gridSize, blockSize](d_u, d_v, edge, corner)

    return d_u.copy_to_host()


def main():
    #Compute exact solution
    exact = np.zeros(shape=[NX,NY], dtype=np.float32)
    for i in range(NX):
        for j in range(NY):
            exact[i,j]= sin(i*PI/(NX-1)) * sinh(j*PI/(NY-1))/sinh(PI)

    #parallel iteration results
    u = np.zeros(shape=[NX,NY], dtype=np.float32)
    for i in range(NX):
        u[i,NX-1]= sin(i*PI/(NX-1)) #boundary conditions
    if STENCIL_POINTS == 5:
        edge, corner = 0.25, 0
    elif STENCIL_POINTS == 9:
        edge, corner = 0.20, 0.05
    else:
        print("Supported values of STENCIL_POINTS: {5,9}")
        return
    u = update(u, iters, edge, corner)

    error = np.max(np.abs(u-exact))
    print("NX = %d, iters = %d => max error: %5.2e"  %(NX, iters, error))

    xvals = np.linspace(0., 1.0, NX)
    yvals = np.linspace(0., 1.0, NY)
    X,Y = np.meshgrid(xvals, yvals)
    levels = [0.025, 0.1, 0.25, 0.50, 0.75]
    plt.contourf(X,Y,exact.T, levels = levels)
    plt.contour(X,Y,u.T, levels = levels,
        colors = 'r', linewidths = 4)
    plt.axis([0,1,0,1])
    plt.show()

if __name__ == '__main__':
    main()

# %%
