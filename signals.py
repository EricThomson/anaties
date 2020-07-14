"""
This is partly adapted from code in the scipy cookbook as well as the filtfilt docs:
   https://scipy-cookbook.readthedocs.io/items/FiltFilt.html 
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
"""

import numpy as np
#from numpy.random import randn
from scipy import signal
#from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
import matplotlib.pyplot as plt 



#%% create a filter
# different numpy windows: 
#    https://numpy.org/doc/stable/reference/routines.window.html
# Each window has a good page of its own
# hanning: cosine-shape
# blackman: more narrowly peak
# bartlett: triangle
# flat: uniform density
#
# At edges can pad or use gustafsson's method. The latter seems better 
# so I am just making gustafsson's the default. 
#
#
def smooth(signal_orig, filter_type = 'hanning', filter_width = 11, plot_on = True):
    """ 
    smooth a 1d signal using filtfilt to have zero phase distortion
    
    Inputs:
        signal: numpy array
        filter_type: string ('flat', 'hanning', 'bartlett', 'blackman')
        filter_width: int (wider is more smooth)
    Outputs
        signal_smoothed: signal after being smoothed 
        filter
        
    Notes: uses gustaffson's method to handle edge artifacts
    """
    if filter_type == 'flat':
        filt_numer = np.ones(filter_width)
    elif filter_type == 'hanning':
        filt_numer = np.hanning(filter_width)
    elif filter_type == 'bartlett':
        filt_numer = np.bartlett(filter_width)
    elif filter_type == 'blackman':
        filt_numer = np.blackman(filter_width)
    filt_denom = np.sum(filt_numer)
    signal_smoothed = signal.filtfilt(filt_numer, filt_denom, 
                                      signal_orig, method = "gust") #pad

    if plot_on:
        # plot filter
        plt.figure(f'{filter_type} filter') 
        plt.plot(filt_numer/filt_denom)
        # plot signal
        plt.figure('signal', figsize=(10,5))
        plt.plot(signal_orig, 'b', label = 'noisy signal', linewidth = 1)
        plt.plot(signal_smoothed, 'k')
        plt.xlim(0, len(signal_smoothed))
        plt.grid(True)

    return signal_smoothed, filt_numer/filt_denom


#%%
if __name__ == '__main__':
    std = 0.4
    t = np.linspace(-1, 1, 201)
    pure_signal = (np.sin(2 * np.pi * 0.75 * t*(1-t) + 2.1) + 
                   0.1*np.sin(2 * np.pi * 1.25 * t + 1) +
                   0.18*np.cos(2 * np.pi * 3.85 * t))
    noisy_signal = pure_signal + np.random.normal(loc=0, scale = std, size = t.shape) 
    
    filter_width = 13
    smoothed_signal, _ = smooth(noisy_signal, 'hanning', filter_width = 13, plot_on = True)

    