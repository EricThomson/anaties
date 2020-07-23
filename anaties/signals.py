"""
signal processing branch of anaties package

https://github.com/EricThomson/anaties
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
import scipy.fftpack as fftpack
import scipy.signal.windows as windows
from scipy.io import wavfile


#%%
def smooth(data, window_type = 'hann', filter_width = 11, sigma = 2, plot_on = 1):
    """ 
    Smooth a 1d data with moving window (uses filtfilt to have zero phase distortion)
        
    Inputs:
        signal: numpy array
        window_type ('hanning'): string ('boxcar', 'gaussian', 'hanning', 'bartlett', 'blackman')
        filter_width (11): int (wider is more smooth) odd is ideal
        sigma (2.): scalar std deviation only used for gaussian 
        plot_on (1): int determines plotting. 0 none, 1 plot signal, 2: also plot filter
    Outputs
        data_smoothed: signal after being smoothed 
        filter_window: the window used for smoothing
        
    Notes: 
        Uses gustaffson's method to handle edge artifacts
        Currently accepted window_type options:
            hann (default) - cosine bump filter_width is only param
            blackman - more narrowly peaked bump than hann
            boxcar - flat-top of length filter_width
            bartlett - triangle
            gaussian - sigma determines width


    """
    if window_type == 'boxcar':
        filter_window = windows.boxcar(filter_width)
    elif window_type == 'hann':
        filter_window = windows.hann(filter_width)
    elif window_type == 'bartlett':
        filter_window = windows.bartlett(filter_width)
    elif window_type == 'blackman':
        filter_window = windows.blackman(filter_width)
    elif window_type == 'gaussian':
        filter_window = windows.gaussian(filter_width, sigma)
    filter_window = filter_window/np.sum(filter_window)
    data_smoothed = signal.filtfilt(filter_window, 1, 
                                      data, method = "gust") #pad

    if plot_on:
        if plot_on > 1:
            plt.plot(filter_window)
            plt.title(f'{window_type} filter') 
        plt.figure('signal', figsize=(10,5))
        plt.plot(data, color = (0.7, 0.7, 0.7), label = 'noisy signal', linewidth = 1)
        plt.plot(data_smoothed, color = 'r', label = 'smoothed signal')
        plt.xlim(0, len(data_smoothed))
        plt.xlabel('sample')
        plt.grid(True)
        plt.legend()

    return data_smoothed, filter_window


def fft(data, sampling_period, include_neg = False, freq_limits = None, plot_on = 1):
    """ 
    Calculates fft, and power spectrum, of 1d data
    
    Inputs:
        data: numpy array
        sampling_period (float): time between samples
        include_neg (bool): include negative frequencies in result?
        freq_limits (2-elt array-like): low and high frequencies used only for plotting (None)
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
        first_ind, last_ind = ind_limits(frequencies, freq_limits) 
        plt.figure('power')
        plt.plot(frequencies[first_ind: last_ind], 
                 power_spectrum[first_ind: last_ind], 
                 color = (0.4, 0.4, 0.4),
                 linewidth = 0.75)
        plt.yscale('log')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel('Frequency')
        plt.ylabel('log(Power)')
        
    return data_fft, power_spectrum, frequencies


def notch_filter(data, notch_frequency, sampling_frequency, quality_factor = 35., plot_on = 1):
    """
    Apply a notch filter at notch_frequency to 1d data (can remove 60Hz for instance)
    
    Inputs:
        data (1d numpy array)
        notch_frequency: the frequency you want removed
        sampling_frequency: frequency (Hz) at which data was sampled
        quality_factor (float): sets bandwidth of notch filter (35)
        plot_on (int): 0 to not plot, 1 to plot filter, original, and filtered signals
        
    Outputs:
        data_filtered (1d numpy array) -- same size as data, but filtered
        b: numerator filter coeffecient array
        a: denominator filter coefficient array
    """

    b, a = signal.iirnotch(notch_frequency, quality_factor, sampling_frequency)
    data_filtered = signal.filtfilt(b, a, data)
    
    if plot_on: 
        # Frequency response
        freq, h = signal.freqz(b, a, fs = sampling_frequency)
        plt.figure('notch')
        plt.subplot(311)
        plt.plot(freq, 20*np.log10(abs(h)))
        plt.autoscale(enable=True, axis='x', tight=True)
    
        # Original signal and filtered version of signal
        plt.subplot(312)
        plt.plot(data, color = (0.2, 0.2, 0.2), linewidth = 1)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.subplot(313)
        plt.plot(data_filtered, color = (0.2, 0.2, 0.2), linewidth = 1)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        
    return data_filtered, b, a


def spectrogram(data, 
                sampling_rate, 
                segment_length = 1024, 
                segment_overlap = 512, 
                window = 'hann', 
                freq_limits = None,
                colormap = 'inferno',
                plot_on = 0):
    """ 
    Get/plot spectrogram of signa -- wrapper for scipy.spectrogram
    
    Inputs:
        data: numpy array
        sampling_freq (float): sampling rate (Hz)
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512)
        window (string): type of window to apply to each segment to make it periodic
        freq_limits (2-elt array-like): low and high frequencies used only for plotting (None)
        colormap (string): colormap (inferno) (see also gist_heat, twilight_shifted, jet, ocean, bone)
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
        print("Plotting spectrogram")
        num_samples = len(data)
        sampling_period = 1/sampling_rate
        duration = num_samples*sampling_period
        times = np.linspace(0, duration, num_samples)
        fig, axs = plt.subplots(2,1, figsize = (12,10), sharex = True)
        axs[0].plot(times, data, color = (0.5, 0.5, 0.5), linewidth = 0.5)
        axs[0].autoscale(enable=True, axis='x', tight=True)
        first_ind, last_ind = ind_limits(freqs, freq_limits)          
        axs[1].pcolormesh(time_bins, 
                          freqs[first_ind:last_ind], 
                          10*np.log10(spect[first_ind: last_ind,:]), cmap = colormap);
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlabel('t(s)')
        axs[1].autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()

    return spect, freqs, time_bins


def ind_limits(data, data_limits = None):
    """ 
    Given increasing data, and two data limits (min and max), returns indices 
    such that data is between those limits (inclusive).
    
    inputs:
        data: nondecreasing 1d np array
        data_limits: limits of data you want to select
    outputs:
        first_ind: index of smallest data >= data_limits[0]
        last_ind: index of largest data <= data_limits[1]
    
    To do:
        Add checks for 1d data, increasing data, data_limits.
    """
    first_ind = 0
    last_ind = -1        
    if data_limits is not None:
        inds = np.where((data >= data_limits[0]) & (data <= data_limits[1]))[0]
        first_ind = inds[0]
        last_ind = inds[-1]
    return first_ind, last_ind

#%%  run some tests
if __name__ == '__main__':
    plt.close('all')
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
    smoothed_signal, gauss_window = smooth(noisy_signal, 
                                           window_type = window, 
                                           filter_width = 13, 
                                           sigma = 3,
                                           plot_on = 1)
    plt.title(f'signals.smooth test with {window} filter')
    plt.show()


    """
    Test fft
    """
    f1 = 20
    f2 = 33
    num_points = 600   # Number of points
    samp_pd = 0.01  # sampling period
    x = np.linspace(0.0, num_points*samp_pd, num_points)
    y = np.sin(f1 * 2.0*np.pi*x) + 0.5*np.sin(f2 * 2.0*np.pi*x)
    full_fft, power_spec, freqs = fft(y, 
                                      samp_pd, 
                                      include_neg = False, 
                                      freq_limits = [5, 50], 
                                      plot_on = 1)
    plt.title('signals.fft test')
    plt.show()
    
    
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
    spectrogram(data, 
                sample_rate, 
                segment_length = 1024, 
                segment_overlap = 512, 
                window = 'hann', 
                freq_limits = [300, 15_000],
                plot_on = 1)
    plt.suptitle('signals.spectrogram test', y = 1)
    plt.show()

    

    """
    Test notch filter
    """
    f1 = 17
    f2 = 60
    notch_frequency = 60
    sampling_frequency = 1000
    duration = 1
    t = np.linspace(0.0, duration, duration*sampling_frequency)
    data = np.sin(f1 * 2.0*np.pi*t) + np.sin(f2 * 2.0*np.pi*t) 
    filtered_data, b, a = notch_filter(data, 
                                       notch_frequency, 
                                       sampling_frequency, 
                                       quality_factor = 35., 
                                       plot_on = 1)
    plt.suptitle('signals.notch filter test', y = 1)
    plt.show()





    # Tests done