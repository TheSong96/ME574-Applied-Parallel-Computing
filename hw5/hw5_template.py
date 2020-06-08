from  numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft

def cupy_fft():

    """
    Modify the code below so it calls the cupy fft function
    """

    pts = 1000
    L = 100
    w0 = 2.0 * np.pi/L
    n1, n2, n3 = 10.0, 20.0, 30.0
    a1, a2, a3 = 1., 2., 3.

    #create signal data with 3 frequency components
    x = np.linspace(0,L,pts)
    y1 = a1*np.cos(n1*w0*x)
    y2 = a2*np.sin(n2*w0*x)
    y3 = a3*np.sin(n3*w0*x)
    y = y1 + y2 + y3

    #create signal including only 2 components
    y12 = y1 + y2

    #analytic derivative of signal
    dy = w0*(-n1*a1*np.sin(n1*w0*x)
            +n2*a2*np.cos(n2*w0*x)
            +n3*a3*np.cos(n3*w0*x) )

    #use fft.fftfreq to get frequency array corresponding to number of sample points
    freqs = fftfreq(pts)
    print(f"freqs = {freqs}")
    #compute number of cycles and radians in sample window for each frequency
    nwaves = freqs*pts
    nwaves_2pi = w0*nwaves

    # compute the fft of the full signal
    fft_vals = fft(y)

    #mask the negative frequencies
    mask = freqs>0
    #double count at positive frequencies
    fft_theo = 2.0 * np.abs(fft_vals/pts)
    #plot fft of signal
    plt.xlim((0,50))
    plt.xlabel('cycles in window')
    plt.ylabel('original amplitude')
    plt.plot(nwaves[mask], fft_theo[mask])
    plt.show()

    #create a copy of the original fft to be used for filtering
    fft_new = np.copy(fft_vals)
    #filter out y3 by setting corr. frequency component(s) to zero
    fft_new[np.abs(nwaves)==n3] = 0.
    #plot fft of filtered signal
    plt.xlim((0,50))
    plt.xlabel('cycles in window')
    plt.ylabel('filtered amplitude')
    plt.plot(nwaves[mask], 2.0*np.abs(fft_new[mask]/pts))
    plt.show()

    #invert the filtered fft with numpy.fft.ifft
    filt_data = np.real(ifft(fft_new))
    #plot filtered data and compare with y12
    plt.plot(x,y12, label='original signal')
    plt.plot(x,filt_data, label='filtered signal')
    plt.xlim((0,50))
    plt.legend()
    plt.show()

    #multiply fft by 2*pi*sqrt(-1)*frequency to get fft of derivative
    dy_fft = 1.0j*nwaves_2pi*fft_vals
    #invert to reconstruct sampled values of derivative
    dy_recon = np.real(ifft(dy_fft))
    #plot reconstructed derivative and compare with analuytical version
    plt.plot(x,dy,label='exact derivative')
    plt.plot(x,dy_recon, label='fft derivative')
    plt.xlim((0,50))
    plt.legend()
    plt.show()

def cupy_filter():

    """
    Implement code below to:
    Create noise consisting of an array of pts random values chosen from a uniform distribution over the interval [−3,3]
    Create a noisy signal by adding noise to the original signal: y_n = y + noise
    Compute and plot the frequency content of the noisy signal.
    Create and apply an appropriate filter to suppress noise in the frequency domain.
    Invert the filtered fft to obtain a "denoised signal".
    Plot and compare the original, noisy, and denoised signals.
    """

    pass

def cupy_eig(mat):
    """
    Returns the eigen values of square matrix mat
    """
    pass

def cupy_J(n):
    """
    Constructs the nxn matrix J (as described in q2)
    Returns J, leading eigenvalue of J, and the associated eigenvector
    """
    pass

def rand_mat_gauss(n):
    """
    Returns nxn size array of random numbers sampled from a normal distribution with a mean of 0 and standard deviation of 1
    (Uses cupy to generate the random matrix)
    """
    pass

def rand_mat_plusminus(n):
    """
    Returns nxn size array of random numbers uniformly distributed on the 2-value set {-1,1}
    OPTIONAL
    """

    pass

def rand_mat_uniform(n):
    """
    Returns nxn size array of random numbers uniformly distrubted on the range [-1,1]
    OPTIONAL
    """

    pass

