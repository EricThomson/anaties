"""
For code under construction

Currently working on FFT
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

#%% Following from: https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
# Not great it doesn't discuss how they got frequencies for x-axis
# or how they went from fft to the power spectrum (why 2/N, why absolute value
# and not square of output?
f1 = 20
f2 = 33
# Number of sample points
N = 600
# sampling period
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(f1 * 2.0*np.pi*x) + 0.5*np.sin(f2 * 2.0*np.pi*x)

yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.figure('signal')
plt.plot(x,y)

plt.figure('fft power spectrum')
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

#%% Following is adapted from the generally excellent:
# https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/

import scipy.fftpack as fftpack

#%%

sig_fft = fftpack.fft(y)
power_spec = np.abs(sig_fft)**2
# fft gets you the frequencies for your fft back, given
# the number of frequencies, and sampling period/spacing
power_spec_freq = fftpack.fftfreq(len(power_spec), T)

# Note this includes negative frequencies, let's cut those
nonneg_inds = power_spec_freq >= 0

plt.plot(power_spec_freq[nonneg_inds], 
         power_spec[nonneg_inds],
         'b')
plt.yscale('log')
plt.xlim(0,100)
plt.axvline(f1)
plt.axvline(f2)