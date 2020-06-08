import numpy as np
import matplotlib.pyplot as plt
from shared import *


def diff(t):
    rst = t * np.cos(np.pi * t) - np.sin(np.pi * t)
    # print(rst)
    return rst


def test_diff():
    t = np.linspace(0, 10, 101)
    y = diff(t)

    print(t.shape, y.shape)

    plt.plot(t, y)
    # plt.show()
    plt.savefig("eign_solution.png")


def run_shared(y):
    derivative = -nth_deriv_shared(y, order=2, rad=1)
    # print(derivative.shape)
    return derivative


def run_serial(y):
    derivative = -nth_deriv_serial(y, order=2)
    return derivative


def run_parallel(y):
    derivative = -nth_deriv_parallel(y, order=2)
    return derivative


if __name__ == "__main__":
    t = np.linspace(0, 1, 101)
    y = np.sin(t).astype(np.float32)
    derivative1 = run_shared(y)
    derivative2 = run_serial(y)
    derivative3 = run_parallel(y)
    # print(derivative1)
    # print(derivative2)
    # print(derivative3)
    # print(np.equal(derivative1, derivative2))
    assert (derivative1 == derivative2).all()
    assert (derivative1 == derivative3).all()
    assert np.equal(derivative2, derivative3).all()

