'''
Consider ways to sum the results from different threads
1) Include a register variable to collect the results
2) Create a global variable to collect the results
3) Something else?
'''

import numpy as np

#serial array sum
def serial_sum(u):
	n = u.shape[0]
	accum = 0
	for i in range(N):
		accum += u[i]
	return accum

N = 128
u = np.ones(N) # 1D array
res = serial_sum(u)
print('serial result = ',res)


from numba import cuda, float32, float64

TPB =  16

# # first try - where to definite/initialize accum?
# @cuda.jit
# def sum_kernel(d_u):
# 	n = d_u.shape[0]
# 	i = cuda.grid(1)
# 	accum = 0
# 	accum += d_u[i]
# 	print("Thread result is ", accum)

# def nu_sum(u):
# 	n = u.shape[0]
# 	d_u = cuda.to_device(u)
# 	blocks = (n+TPB-1)//TPB
# 	threads = TPB
# 	#accum = 0
# 	sum_kernel[blocks, threads](d_u)
	
# nu_sum(u)
# #print("first parallel try gives ", accum)

# second try
# @cuda.jit
# def sum_kernel(accum, d_u):
# 	n = d_u.shape[0]
# 	i = cuda.grid(1)
# 	accum += u[i]

# def nu_sum(u):
# 	n = u.shape[0]
# 	accum = 0
# 	d_u = cuda.to_device(u)
# 	blocks = (n+TPB-1)//TPB
# 	threads = TPB
# 	sum_kernel[blocks, threads](accum,d_u)
# 	print("second parallel try gives ", accum)

# nu_sum(u)

# third try
# @cuda.jit
# def sum_kernel(d_accum, d_u):
# 	n = d_u.shape[0]
# 	i = cuda.grid(1)
# 	if i>=n:
# 		return
# 	d_accum[0] += d_u[i]

# def nu_sum(u):
# 	n = u.shape[0]
# 	accum = np.zeros(1)
# 	d_accum = cuda.to_device(accum)
# 	d_u = cuda.to_device(u)
# 	blocks = (n+TPB-1)//TPB
# 	threads = TPB
# 	sum_kernel[threads,blocks](d_accum,d_u)
# 	accum = d_accum.copy_to_host()
# 	return accum[0]

# res = nu_sum(u)
# print("Third parallel try gives = ", res)

#fourth try: Atomic operation
# @cuda.jit
# def sum_kernel(d_accum, d_u):
# 	n = d_u.shape[0]
# 	i = cuda.grid(1)
# 	if i>+n:
# 		return
# 	cuda.atomic.add(d_accum,0,d_u[i])
# 	# discuss serialization...

# def nu_sum(u):
# 	n = u.shape[0]
# 	accum = np.zeros(1)
# 	d_accum = cuda.to_device(accum)
# 	d_u = cuda.to_device(u)
# 	blocks = (n+TPB-1)//TPB
# 	threads = TPB
# 	sum_kernel[threads,blocks](d_accum,d_u)
# 	accum = d_accum.copy_to_host()
# 	return accum[0]

# res = nu_sum(u)
# print("Fourth parallel try gives = ", res)

# How many threads read, augmented, and wrote to accum[0]?

#Fifth parallel version
# @cuda.jit
# def sum_kernel(d_accum, d_u):
# 	n = d_u.shape[0]
# 	nsh = cuda.blockDim.x
# 	u_sh = cuda.shared.array(TPB, dtype=float64)
# 	tx = cuda.threadIdx.x
# 	i = cuda.grid(1)
# 	if i>+n:
# 		return
# 	u_sh[tx] = d_u[i]
# 	cuda.syncthreads()
# 	if tx == 0:
# 		block_sum = 0
# 		for i in range(TPB):
# 			block_sum += u_sh[tx]
# 		cuda.atomic.add(d_accum,0,block_sum)
# 	# again discuss serialization...

# def nu_sum(u):
# 	n = u.shape[0]
# 	accum = np.zeros(1)
# 	d_accum = cuda.to_device(accum)
# 	d_u = cuda.to_device(u)
# 	blocks = (n+TPB-1)//TPB
# 	threads = TPB
# 	sum_kernel[threads,blocks](d_accum,d_u)
# 	accum = d_accum.copy_to_host()
# 	return accum[0]

# res = nu_sum(u)
# print("Fifth parallel try gives = ", res)

# def main():
# 	u = np.ones(N) # 1D array
# 	res = serial_sum(u)
# 	print('res = ',res)

# 	res = nu_sum(u)
# 	print('parallel: res = ',res)

# if __name__ == '__main__':
# 	main()


# numba's built-in cuda reduction
# @cuda.reduce
# def sum_reduce(a, b):
#     return a + b

# res = sum_reduce(u)
# print("Reduction result is ", res)
