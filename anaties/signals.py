"""
This is partly adapted from code in the scipy cookbook as well as the filtfilt docs:
   https://scipy-cookbook.readthedocs.io/items/FiltFilt.html 
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
import scipy.fftpack as fftpack
import scipy.signal.windows as windows
from scipy.io import wavfile


#%%
def smooth(data, filter_type = 'hann', filter_width = 11, sigma = 2, plot_on = 1):
    """ 
    smooth a 1d signal using filtfilt to have zero phase distortion
    filter type options:
        hann (default) - cosine bump filter_width is only param
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
        data_smoothed: signal after being smoothed 
        filter
        
    Notes: uses gustaffson's method to handle edge artifacts
    """
    if filter_type == 'boxcar':
        filt_numer = windows.boxcar(filter_width)
    elif filter_type == 'hann':
        filt_numer = windows.hann(filter_width)
    elif filter_type == 'bartlett':
        filt_numer = windows.bartlett(filter_width)
    elif filter_type == 'blackman':
        filt_numer = windows.blackman(filter_width)
    elif filter_type == 'gaussian':
        filt_numer = windows.gaussian(filter_width, sigma)
    filt_denom = np.sum(filt_numer)
    data_smoothed = signal.filtfilt(filt_numer, filt_denom, 
                                      data, method = "gust") #pad

    if plot_on:
        if plot_on > 1:
            plt.plot(filt_numer/filt_denom)
            plt.title(f'{filter_type} filter') 
        plt.figure('signal', figsize=(10,5))
        plt.plot(data, color = (0.7, 0.7, 0.7), label = 'noisy signal', linewidth = 1)
        plt.plot(data_smoothed, color = 'r', label = 'smoothed signal')
        plt.xlim(0, len(data_smoothed))
        plt.xlabel('sample')
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()

    return data_smoothed, filt_numer/filt_denom


def fft(data, sampling_period, include_neg = False, plot_on = 1):
    """ 
    Calculates fft, and power spectrum
    
    Inputs:
        data: numpy array
        sampling_period (float): time between samples
        include_neg (bool): include negative frequencies in result?
        plot_on (int): determines plotting (0 no, 1 yes)
        
    Outputs:
        data_fft: the full fft from fftpack
        power_spectrum: the amplitude squared at each frequency
        frequencies: corresponding frequencies of power spectrum 
    
    Adapted from:
        https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/
    """
    data_fft = fftpack.fft(data)
    power_spectrum = np.abs(data_fft)**2
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
        
    return data_fft, power_spectrum, frequencies


def spectrogram(data, 
                sampling_rate, 
                segment_length = 1024, 
                segment_overlap = 512, 
                window = 'hann', 
                plot_on = 1):
    """ 
    Get/plot spectrogram of signa -- wrapper for scipy.spectrogram
    
    Inputs:
        data: numpy array
        sampling_freq (float): sampling rate (Hz)
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512)
        window (string): type of window to apply to each segment to make it periodic
        plot_on (int): 0 for no plotting, 1 to plot signal/spectrogram (0)
    
    Returns
        spectrogram (num_freqs x num_time_points)
        freqs (array of frequencies)
        time_bins (time bin centers)

    Notes:
        - To plot use pcolormesh and 10*log10(spectrogram) else it will look weird.
        - Windowing is not for smoothing, but to extract the data for the short-time FFT --
           the segment_length window makes the data segment quasi-periodic (wraps around
           values as the window drops to zero). This makes the FFT behave. Do not use
           boxcar I would stick with hann or similar.
    """
    if data.ndim > 1:
        # if array is (n,1) that is still 2d and will break spectrogram. flatten it
        data = data.flatten()
        
    freqs, time_bins, spect = signal.spectrogram(data, 
                                             fs = sampling_rate,
                                             nperseg = segment_length,
                                             noverlap = segment_overlap,
                                             window = window)
    if plot_on:
        colormap ='inferno' 
        num_samples = len(data)
        sampling_period = 1/sampling_rate
        duration = num_samples*sampling_period
        times = np.linspace(0, duration, num_samples)
        fig, axs = plt.subplots(2,1, figsize = (12,10), sharex = True)
        axs[0].plot(times, data, color = (0.5, 0.5, 0.5), linewidth = 0.5)
        axs[0].autoscale(enable=True, axis='x', tight=True)
        
        axs[1].pcolormesh(time_bins, freqs, 10*np.log10(spect), cmap = colormap);
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlabel('t(s)')
        axs[1].autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()

    
    return spectrogram, freqs, time_bins
    
    

#%%  run some tests
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
                                sigma = 3,
                                plot_on = 2)
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
    
    
    """
    Test spectrogram
    """
    # First extract some sample audio data to analyze
    wav_path = r'../data/songbirds.wav'
    sample_rate, data_full = wavfile.read(wav_path)
    start_ind = 3_450_000
    num_samples = 300_000 #1_500_000
    data = data_full[start_ind: start_ind+num_samples, 0]
    segment_length = 1024
    segment_overlap = segment_length//2
    spectrogram(data, sample_rate, segment_length = 1024, 
                segment_overlap = 512, window = 'hann', plot_on = 1)
    plt.suptitle('signals.spectrogram test')
    
    
    
    
    # Test s done