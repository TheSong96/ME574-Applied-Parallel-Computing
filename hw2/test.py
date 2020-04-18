import numpy as np
import matplotlib.pyplot as plt
from time import time #import timing function
from numba import cuda
import seaborn ; seaborn.set()
N = 640000

def time_main():
    start_all = time() #start overall timer
    x = np.linspace(0, 1, N, endpoint=True)
    from map_serial import sArray
    start = time() #start timer for serial execution
    f = sArray(x)
    end = time() #stop timer for serial execution
    elapsed_serial = end - start #compute serial runtime
    print("--- Serial timing: %0.4f seconds ---" % elapsed_serial)
    
    from map_parallel import sArray #import parallel version of sArray
    for i in range(2):
        start = time() #start timer for parallel execution
        fpar = sArray(x)
        end = time() #stop timer for parallel execution
        elapsed = end - start #compute parallel runtime
        
        print("--- Parallel timing #%d: %3.4f seconds ---" % (i,elapsed))
    print("--- Loop acceleration estimate: %dx ---" % (elapsed_serial//elapsed))
    end_all = time() #end overall timer                                       
    elapsed = end_all - start_all #evaluate overall runtime
    print("--- Total time: %3.4f seconds ---" % elapsed)
    print("--- Total acceleration estimate: %3.4fx ---" % ((3*elapsed_serial)/(elapsed)))

def main():
    N = 64
    x = np.linspace (0,1,N, dtype = np.float32)

    from serial import sArray
    f = sArray (x)

    plt.plot (x, f, 'o')
    plt.show ()

if __name__ == '__main__ ':
    main()