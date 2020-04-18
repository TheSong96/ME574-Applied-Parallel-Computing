import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt

def gpu_total_memory():
	'''
	Query the GPU's properties via Numba to obtain the total memory of the device.
	'''
	pass

def gpu_compute_capability():
	'''
	Query the GPU's properties via Numba to obtain the compute capability of the device.
	'''
	pass

def gpu_name():
	'''
	Query the GPU's properties via Numba to obtain the name of the device.
	'''
	pass

def max_float64s():
	'''
	Compute the maximum number of 64-bit floats that can be stored on the GPU
	'''
	pass

def map_64():
	'''
	Execute the map app modified to use 64-bit floats
	'''
	from map_parallel import sArray

#@cuda.jit(device = True)
def f(x, r):
	'''
	Execute 1 iteration of the logistic map
	'''
	return r*x*(1 - x)

#@cuda.jit
def logistic_map_kernel(ss, r, x, transient, steady):
	'''
	Kernel for parallel iteration of logistic map

	Arguments:
		ss: 2D numpy device array to store steady state iterates for each value of r
		r: 1D  numpy device array of parameter values
		x: float initial value
		transient: int number of iterations before storing results
		steady: int number of iterations to store
	'''
	pass

def parallel_logistic_map(r, x, transient, steady):
	'''
	Parallel iteration of the logistic map

	Arguments:
		r: 1D numpy array of float64 parameter values
		x: float initial value
		transient: int number of iterations before storing results
		steady: int number of iterations to store
	Return:
		2D numpy array of steady iterates for each entry in r
	'''
	pass

#@cuda.jit(device = True)
def iteration_count(cx, cy, dist, itrs):
	'''
	Computed number of Mandelbrot iterations

	Arguments:
		cx, cy: float64 parameter values
		dist: float64 escape threshold
		itrs: int iteration count limit
	'''
	pass

#@cuda.jit
def mandelbrot_kernel(out, cx, cy, dist, itrs):
	'''
	Kernel for parallel computation of Mandelbrot iteration counts

	Arguments:
		out: 2D numpy device array for storing computed iteration counts
		cx, cy: 1D numpy device arrays of parameter values
		dist: float64 escape threshold
		itrs: int iteration count limit
	'''
	pass

def parallel_mandelbrot(cx, cy, dist, itrs):
	'''
	Parallel computation of Mandelbrot iteration counts

	Arguments:
		cx, cy: 1D numpy arrays of parameter values
		dist: float64 escape threshold
		itrs: int iteration count limit
	Return:
		2D numpy array of iteration counts
	'''
	pass

if __name__ == "__main__":
	
	#Problem 1
	print("GPU memory in GB: ", gpu_total_memory()/1024**3)
	print("Compute capability (Major, Minor): ", gpu_compute_capability())
	print("GPU Model Name: ", gpu_name())
	print("Max float64 count: ", max_float64s())

	#PASTE YOUR OUTPUT HERE#

	#Problem 2
	map_64()

	#PASTE YOUR ERROR MESSAGES HERE#

	#Problem 3

	#Problem 4
