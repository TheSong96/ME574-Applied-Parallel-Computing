# Sample code illustrating use of numpy.fft for filtering and differentiating

from  numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft

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

#multiply fft by 2*pi*sqrt(-1)*frequeny to get fft of derivative
dy_fft = 1.0j*nwaves_2pi*fft_vals
#invert to reconstruct sampled values of derivative
dy_recon = np.real(ifft(dy_fft))
#plot reconstructed derivative and compare with analuytical version
plt.plot(x,dy,label='exact derivative')
plt.plot(x,dy_recon, label='fft derivative')
plt.xlim((0,50))
plt.legend()
plt.show()