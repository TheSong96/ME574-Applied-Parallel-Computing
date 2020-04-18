import numpy as np
import matplotlib.pyplot as plt
from map_parallel import sArray

def main(plot = False):
	N = 128
	x = np.linspace(0,1,N,dtype = np.float32)
	y = sArray(x)

	if plot:
		plt.plot(x,y)
		plt.show()

	return y

if __name__ == "__main__":
	main(plot = True)
