"""
This is partly adapted from code in the scipy cookbook as well as the filtfilt docs:
   https://scipy-cookbook.readthedocs.io/items/FiltFilt.html 
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
"""

import numpy as np
from scipy import signal
import scipy.fftpack as fftpack
import scipy.signal.windows as windows
import matplotlib.pyplot as plt 


#%% create a filter
# different numpy windows: 
#    https://numpy.org/doc/stable/reference/routines.window.html
# Each window has a good page of its own
# hanning: cosine-shape
# blackman: more narrowly peaked
# bartlett: triangle
# flat: uniform density
# gaussian: gaussian (requires sigma (std) defaults to 1)
#
# At edges can pad or use gustafsson's method. The latter seems better 
# so I am just making gustafsson's the default. 
#
#
def smooth(signal_orig, filter_type = 'hanning', filter_width = 11, sigma = 2, plot_on = 1):
    """ 
    smooth a 1d signal using filtfilt to have zero phase distortion
    filter type options:
        hanning (default) - cosine bump filter_width is only param
        blackman - more narrowly peaked bump than hanning
        gaussian - sigma determines width
        boxcar - flat-top of length filter_width
        bartlett - triangle
    
    Inputs:
        signal: numpy array
        filter_type ('hanning'): string ('boxcar', 'gaussian', 'hanning', 'bartlett', 'blackman')
        filter_width (11): int (wider is more smooth) odd is ideal
        sigma (2.): scalar std deviation only used for gaussian 
        plot_on (1): int determines plotting. 0 none, 1 plot signal, 2: also plot filter
    Outputs
        signal_smoothed: signal after being smoothed 
        filter
        
    Notes: uses gustaffson's method to handle edge artifacts
    """
    if filter_type == 'boxcar':
        filt_numer = windows.boxcar(filter_width)
    elif filter_type == 'hanning':
        filt_numer = windows.hann(filter_width)
    elif filter_type == 'bartlett':
        filt_numer = windows.bartlett(filter_width)
    elif filter_type == 'blackman':
        filt_numer = windows.blackman(filter_width)
    elif filter_type == 'gaussian':
        filt_numer = windows.gaussian(filter_width, sigma)
    filt_denom = np.sum(filt_numer)
    signal_smoothed = signal.filtfilt(filt_numer, filt_denom, 
                                      signal_orig, method = "gust") #pad

    if plot_on:
        if plot_on > 1:
            plt.figure(f'{filter_type} filter') 
            plt.plot(filt_numer/filt_denom)
        plt.figure('signal', figsize=(10,5))
        plt.plot(signal_orig, color = (0.7, 0.7, 0.7), label = 'noisy signal', linewidth = 1)
        plt.plot(signal_smoothed, color = 'r', label = 'smoothed signal')
        plt.xlim(0, len(signal_smoothed))
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()

    return signal_smoothed, filt_numer/filt_denom


def fft(signal_orig, sampling_period, include_neg = False, plot_on = 1):
    """ 
    Calculates fft, and power spectrum
    
    Inputs:
        signal_orig: numpy array
        sampling_period (float): time between samples
        include_neg (bool): include negative frequencies in result?
        plot_on (int): determines plotting (0 no, 1 yes)
        
    Outputs:
        sig_fft: the full fft from fftpack
        power_spectrum: the amplitude squared at each frequency
        frequencies: corresponding frequencies of power spectrum 
    
    Adapted from:
        https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/
    """
    sig_fft = fftpack.fft(signal_orig)
    power_spectrum = np.abs(sig_fft)**2
    # returns frequencies given signal length and sampling period
    frequencies = fftpack.fftfreq(len(power_spectrum), sampling_period)
    
    if not include_neg:
        pos_inds = frequencies > 0
        frequencies = frequencies[pos_inds]
        power_spectrum = power_spectrum[pos_inds]
    
    if plot_on:
        plt.figure('power')
        plt.plot(frequencies, 
                 power_spectrum)
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('Frequency')
        plt.ylabel('log(Power)')
        
    return sig_fft, power_spectrum, frequencies




#%%
if __name__ == '__main__':
    """
    Test smooth
    """
    std = 0.4
    t = np.linspace(-1, 1, 201)
    pure_signal = (np.sin(2 * np.pi * 0.75 * t*(1-t) + 2.1) + 
                   0.1*np.sin(2 * np.pi * 1.25 * t + 1) +
                   0.18*np.cos(2 * np.pi * 3.85 * t))
    noisy_signal = pure_signal + np.random.normal(loc=0, scale = std, size = t.shape) 
    filter_width = 13
    window = 'gaussian'
    smoothed_signal, _ = smooth(noisy_signal, 
                                filter_type = window, 
                                filter_width = 13, 
                                plot_on = False)
    plt.figure('signal', figsize=(10,5))
    plt.plot(t, noisy_signal, 'k', linewidth = 0.5, label = 'noisy')
    plt.plot(t, smoothed_signal, 'r', label = 'smoothed')
    plt.plot(t, pure_signal, 'g', label = 'original')
    plt.xlim(-1, 1)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(f'signals.smooth test with {window} filter')

    """
    Test fft
    """
    f1 = 20
    f2 = 33
    num_points = 600   # Number of points
    samp_pd = 0.01  # sampling period
    x = np.linspace(0.0, num_points*samp_pd, num_points)
    y = np.sin(f1 * 2.0*np.pi*x) + 0.5*np.sin(f2 * 2.0*np.pi*x)
    _, power_spec, freqs = fft(y, samp_pd, include_neg = False, plot_on = True)
    plt.title('signals.fft test')
    
    # Test spectrogram
    
    

    
    
    # Test 