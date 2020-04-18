import numpy as np

from hw1 import *

n = 10
for i in range(n):
    print("{}th order derivative = {}".format(i, dn_sin(i)))

n = 10
x = np.pi/2
for i in range(1, n):
    print("{}th Taylor Series at x = {} is {}".format(i, x, taylor_sin(x, i)))

ary1 = np.array([1,2,3,4,5,6]).reshape((2,3))
ary2 = np.array([2,2,3,4,5,6]).reshape((2,3))
diff = measure_diff(ary1, ary2)
print(diff)
