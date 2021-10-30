"""
signal processing functions for anaties package

https://github.com/EricThomson/anaties
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
import scipy.signal.windows as windows
from scipy.io import wavfile

from .helpers import ind_limits
# %%


def smooth(data, window_type='hann', filter_width=11, sigma=2, plot_on=1):
    """
    Smooth 1d data with moving window (uses filtfilt to have zero phase distortion)
    Wrapper for scipy.signal.filtfilt
    To do: consider replacing with sosfiltfilt

    Inputs:
        data: numpy array
        window_type ('hann'): string ('boxcar', 'gaussian', 'hann', 'bartlett', 'blackman')
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
                                    data, method="gust")  # pad

    if plot_on:
        if plot_on > 1:
            plt.plot(filter_window)
            plt.title(f'{window_type} filter')
        plt.figure('signal', figsize=(10, 5))
        plt.plot(data, color=(0.7, 0.7, 0.7),
                 label='noisy signal', linewidth=1)
        plt.plot(data_smoothed, color='r', label='smoothed signal')
        plt.xlim(0, len(data_smoothed))
        plt.xlabel('sample')
        plt.grid(True)
        plt.legend()

    return data_smoothed, filter_window


def smooth_rows(data, window_type='hann', filter_width=11, sigma=2):
    """
    Smooth each row of a 2d array: uses smooth() see that for more details
    for the params.

    Inputs:
        data: nxm numpy array (n=num signals, m = num dimensions)
        window_type ('hann'): string ('boxcar', 'gaussian', 'hann', 'bartlett', 'blackman')
        filter_width (11): int (wider is more smooth) odd is ideal
        sigma (2.): scalar std deviation only used for gaussian window
    Outputs
        data_smoothed: nxm array after being smoothed (nxm 2d array)
    """
    nrows, ncols = data.shape
    data_smoothed = []
    for row in range(nrows):
        smoothed_row, _ = smooth(data[row, :],
                                 window_type=window_type,
                                 filter_width=filter_width,
                                 sigma=sigma,
                                 plot_on=False)
        data_smoothed.append(smoothed_row)
    return np.asarray(data_smoothed)


def power_spec(data,
               sampling_frequency,
               segment_length=1024,
               segment_overlap=512,
               window='hann',
               view_range=None,
               scaling='density',
               plot_on=0):
    """
    Calculates power spectral density (default), or power spectrum, of signal
    using Welch's method.
    Wrapper for scipy.signal.welch

    Inputs:
        data: numpy array
        sampling_frequency (float): frequency (Hz) at which signals were sampled
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512) (must be less than segment_length)
        window (string): type of window to apply to each segment to make it periodic ('hann')
        view_range (2-elt array-like): low and high frequencies used only for plotting (None)
        scaling (str): 'density' for power spectral density, 'spectrum' for power spectrum ('density')
        plot_on (int): 0 for no plotting, 1 to plot spectrum (0)

    Outputs:
        ps: the power spectral density, or power spectrum of the signal
        frequencies: frequencies of spectrum

    Note:
        Frequencies range from sampling_frequency/num_points up to sampling_frequency/2,
        in increments of sampling_frequency/num_points.
    """

    frequencies, ps = welch(data.reshape(-1,),
                            fs=sampling_frequency,  # sample rate
                            window='hann',
                            nperseg=segment_length,
                            noverlap=segment_overlap,
                            detrend='constant',
                            average='mean',  # other option is 'median'
                            scaling=scaling)

    if plot_on:
        first_ind, last_ind = ind_limits(frequencies, view_range)
        plt.figure()
        plt.semilogy(frequencies[first_ind: last_ind],
                     ps[first_ind: last_ind],
                     color=(0.4, 0.4, 0.4),
                     linewidth=0.75)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel('Frequency')
        plt.ylabel('log(Power)')

    return ps, frequencies


def bandpass_filter(data, lowcut, highcut, sampling_frequency, order=5, plot_on=0):
    """
    Bandpass filter signal between locut and highcut frequencies
    Uses butterworth second order section.
    Wrapper for scipy.signal.butter
    Adapted from: https://stackoverflow.com/a/48677312/1886357

    Inputs:
        data (1d numpy array)
        lowcut: low cutoff frequency
        highcut: high cutoff frequency
        sampling_frequency: frequency (Hz) at which data was sampled
        order: order of filter (higher is sharper corners) (5)
        plot_on (int): 0 no plot, 1 to plot filter, original, and filtered signals

    Outputs:
        data_filtered (1d numpy array) -- same size as data, but filtered
        butter_sos: butterworth bandpass filter (second order section)

    """
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    butter_sos = signal.butter(
        order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = signal.sosfiltfilt(butter_sos, data)
    if plot_on:
        print("Plotting")
        #  Filter
        w, h = signal.sosfreqz(butter_sos, worN=2000)
        plt.subplot(2, 1, 1)
        plt.plot((sampling_frequency * 0.5 / np.pi) * w, abs(h))
        plt.axvline(lowcut, color='r', linewidth=0.5)
        plt.axvline(highcut, color='r', linewidth=0.5)
        plt.ylabel('Gain')
        plt.xlabel('Frequency')
        plt.autoscale(enable=True, axis='x', tight=True)

        # Data: original and filtered
        plt.subplot(2, 1, 2)
        plt.plot(data, color=(0.7, 0.7, 0.7), linewidth=0.5)
        plt.plot(filtered_data, color='r', linewidth=1)
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.autoscale(enable=True, axis='x', tight=True)

        plt.tight_layout()

    return filtered_data, butter_sos


def notch_filter(data, notch_frequency, sampling_frequency, quality_factor=35., plot_on=1):
    """
    Apply a notch filter at notch_frequency to 1d data (can remove 60Hz for instance)
    Wrapper for scipy.signal.iirnotch

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
    data = data.flatten()

    b, a = signal.iirnotch(notch_frequency, quality_factor, sampling_frequency)
    data_filtered = signal.filtfilt(b, a, data)

    if plot_on:
        # Frequency response
        freq, h = signal.freqz(b, a, fs=sampling_frequency)
        ig, axs = plt.subplots(3, 1, figsize=(12, 10))
        # Filter
        axs[0].plot(freq, 20*np.log10(abs(h)))
        axs[0].autoscale(enable=True, axis='x', tight=True)
        axs[0].set_xlabel('Frequency')
        axs[0].set_ylabel('Power')
        # Original signal and filtered version of signal
        axs[1].plot(data, color=(0.2, 0.2, 0.2), linewidth=1)
        axs[1].autoscale(enable=True, axis='x', tight=True)
        axs[2].plot(data_filtered,
                    color=(0.2, 0.2, 0.2),
                    linewidth=1)
        axs[2].autoscale(enable=True, axis='x', tight=True)
        axs[2].set_xlabel('Sample')
        axs[2].get_shared_x_axes().join(axs[1], axs[2])
        axs[2].get_shared_y_axes().join(axs[1], axs[2])
        plt.suptitle('Notch filtered Data', y=1)
        plt.tight_layout()

    return data_filtered, b, a


def spectrogram(data,
                sampling_frequency,
                segment_length=1024,
                segment_overlap=512,
                window='hann',
                view_range=None,
                all_events=None,
                colormap='inferno',
                notch_frequency=None,
                plot_on=0,
                scaling='density'):
    """
    Get/plot spectrogram of signal
    Wrapper for scipy.spectrogram

    Inputs:
        data: numpy array
        sampling_frequency (float): frequency (Hz) at which samples were acquired
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512) (must be less than segment_length)
        window (string): type of window to apply to each segment to make it periodic
        view_range (2-elt array-like): low and high frequencies used only for plotting (None)
        all_events (list of lists): times to show vertical bands for events, used for plotting
        colormap (string): colormap (inferno) (see also gist_heat, twilight_shifted, jet, ocean, bone)
        notch_frequency (float): if you want to filter out a frequency first (None)
        plot_on (int): 0 for no plotting, 1 to plot signal/spectrogram (0)
        scaling (str): 'density' for power spectral density, 'spectrum' for power spectrum ('density')

    Outputs:
        spectrogram (num_freqs x num_time_points) power spectral density over time
        freqs (array of frequencies): from `sampling_frequency/segment_length` up to `sampling_frequency/2`
        time_bins (time bin centers): can control resolution w/segment_overlap. DeltaT = (segment_length-segment_overlap)/sampling_freq
        axs: axes (None if plot_on is 0)

    Notes:
        - To plot use pcolormesh and 10*log10(spectrogram) otherwise it will look weird.
        - Windowing is not for smoothing, but to extract the data for the short-time FFT --
           the segment_length window makes the data segment quasi-periodic (wraps around
           values as the window drops to zero). This makes the FFT behave. Do not use
           boxcar I would stick with hann or similar.
    To do
        - I had it using different colors for different events but removed this. Maybe add
        this feature back at some point.
    """
    if data.ndim > 1:
        data = data.flatten()

    if notch_frequency is not None:
        data, _, _ = notch_filter(data,
                                  notch_frequency,
                                  sampling_frequency,
                                  plot_on=0)

    freqs, time_bins, spect = signal.spectrogram(data,
                                                 fs=sampling_frequency,
                                                 nperseg=segment_length,
                                                 noverlap=segment_overlap,
                                                 window=window,
                                                 detrend='constant',  # removes mean from each segment
                                                 scaling=scaling)
    if plot_on:
        num_samples = len(data)
        sampling_period = 1/sampling_frequency
        duration = num_samples*sampling_period
        times = np.linspace(0, duration, num_samples)
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        # Plot raw signal
        axs[0].plot(times, data, color=(0.5, 0.5, 0.5), linewidth=0.5)
        axs[0].autoscale(enable=True, axis='x', tight=True)
        # Plot spectrogram
        first_ind, last_ind = ind_limits(freqs, view_range)
        axs[1].pcolormesh(time_bins,
                          freqs[first_ind:last_ind],
                          10*np.log10(spect[first_ind: last_ind, :]), cmap=colormap)
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlabel('t(s)')
        axs[1].autoscale(enable=True, axis='x', tight=True)
        # Plot events
        if all_events is not None:
            for event_ind, events in enumerate(all_events):
                for event in events:
                    axs[0].axvline(x=event, zorder=3, color='k', linewidth=0.5)
                    axs[1].axvline(x=event, zorder=3, color='k', linewidth=0.5)
        plt.tight_layout()
    else:
        axs = None

    return spect, freqs, time_bins, axs


# %%  run some tests
if __name__ == '__main__':
    plt.close('all')
    """
    Test smooth
    """
    print("anaties.signals: testing smooth()...")
    std = 0.4
    t = np.linspace(-1, 1, 201)
    pure_signal = (np.sin(2 * np.pi * 0.75 * t*(1-t) + 2.1) +
                   0.1*np.sin(2 * np.pi * 1.25 * t + 1) +
                   0.18*np.cos(2 * np.pi * 3.85 * t))
    noisy_signal = pure_signal + \
        np.random.normal(loc=0, scale=std, size=t.shape)
    filter_width = 13
    window = 'gaussian'
    smoothed_signal, gauss_window = smooth(noisy_signal,
                                           window_type=window,
                                           filter_width=13,
                                           sigma=3,
                                           plot_on=1)
    plt.title(f'signals.smooth test with {window} filter')
    plt.show()

    """
    Test smooth_rows()
    """
    print("anaties.signals: testing smooth_rows()")
    noisy_array = [pure_signal+np.random.normal(scale=std, size=t.shape),
                   pure_signal+2+np.random.normal(scale=std, size=t.shape),
                   pure_signal-2+np.random.normal(scale=std, size=t.shape)]
    noisy_array = np.asarray(noisy_array)
    smoothed_array = smooth_rows(noisy_array,
                                 window_type='gaussian',
                                 sigma=3,
                                 filter_width=21)
    plt.figure("smooth_rows()")
    plt.plot(noisy_array.T, linewidth=0.5, color=(0.5, 0.5, 0.5))
    plt.plot(smoothed_array.T, linewidth=1, color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title('signals.smoothed_rows test')
    plt.show()

    """
    Test power_spec
    """
    print("\nanaties.signals: testing power_spec()...")
    f1 = 20
    f2 = 33
    num_points = 1000   # Number of points
    samp_pd = 0.001  # sampling period
    samp_freq = 1/samp_pd
    x = np.linspace(0.0, num_points*samp_pd, num_points)
    y = np.sin(f1 * 2.0*np.pi*x) + 0.5*np.sin(f2 * 2.0*np.pi*x)
    spectral_dens, freqs = power_spec(y,
                                      samp_freq,
                                      segment_length=512,
                                      segment_overlap=200,
                                      view_range=[3, 50],
                                      plot_on=1)
    plt.title('signals.power_spec() test')
    plt.axvline(f1, color='r')
    plt.axvline(f2, color='r')
    plt.show()

    """
    Test spectrogram
    """
    # First extract some sample audio data to analyze
    wav_path = r'../data/songbirds.wav'
    sample_rate, data_full = wavfile.read(wav_path)
    start_ind = 3_450_000
    num_samples = 300_000  # 1_500_000
    data = data_full[start_ind: start_ind+num_samples, 0]
    segment_length = 1024
    segment_overlap = segment_length//2
    event1 = [1.86, 5.3]
    event2 = [2.86, 6.3]
    spect, spect_freqs, spect_time_bins, axs = spectrogram(data,
                                                           sample_rate,
                                                           segment_length=1024,
                                                           segment_overlap=512,
                                                           window='hann',
                                                           view_range=[
                                                               300, 15_000],
                                                           all_events=[
                                                               event1, event2],
                                                           plot_on=1)
    plt.suptitle('signals.spectrogram test', y=1)
    plt.show()

    """
    Test notch filter
    """
    print("\nanaties.signals: testing notch_filter()...")
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
                                       quality_factor=35.,
                                       plot_on=1)
    plt.suptitle('signals.notch filter test', y=1)
    plt.show()

    """
    test bandpass filter
    """
    print("\nanaties.signals: testing bandpass_filter()...")
    samp_freq = 1000  # Sample frequency (Hz)
    f1 = 13
    f2 = 27
    f3 = 60
    std = 0.4
    num_points = 3_000   # Number of points
    samp_freq = 2000
    samp_pd = 1/samp_freq  # sampling period
    duration = num_points * samp_pd
    t = np.linspace(0.0, duration, num_points)
    y_pure = np.sin(f1*2.0*np.pi * t) + np.sin(f2*2.0 *
                                               np.pi * t) + np.sin(f3*2.0*np.pi * t)
    y_noisy = y_pure + np.random.normal(loc=0, scale=std, size=y_pure.shape)
    low_cut = 20
    high_cut = 34
    filter_order = 5  # can mess with t his
    filtered_y, sos_filter = bandpass_filter(
        y_noisy, low_cut, high_cut, samp_freq, order=5, plot_on=1)

    print("\nanaties.signals: tests done...")
    # Tests done
"""
signal processing functions for anaties package

https://github.com/EricThomson/anaties
"""

# import scipy.fftpack as fftpack #not used

# sys.path.append(str(Path('.').absolute().parent))
# from anaties.helpers import ind_limits

# %%


def smooth(data, window_type='hann', filter_width=11, sigma=2, plot_on=1):
    """
    Smooth 1d data with moving window (uses filtfilt to have zero phase distortion)
    Wrapper for scipy.signal.filtfilt
    To do: consider replacing with sosfiltfilt

    Inputs:
        data: numpy array
        window_type ('hann'): string ('boxcar', 'gaussian', 'hann', 'bartlett', 'blackman')
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
                                    data, method="gust")  # pad

    if plot_on:
        if plot_on > 1:
            plt.plot(filter_window)
            plt.title(f'{window_type} filter')
        plt.figure('signal', figsize=(10, 5))
        plt.plot(data, color=(0.7, 0.7, 0.7),
                 label='noisy signal', linewidth=1)
        plt.plot(data_smoothed, color='r', label='smoothed signal')
        plt.xlim(0, len(data_smoothed))
        plt.xlabel('sample')
        plt.grid(True)
        plt.legend()

    return data_smoothed, filter_window


def smooth_rows(data, window_type='hann', filter_width=11, sigma=2):
    """
    Smooth each row of a 2d array: uses smooth() see that for more details
    for the params.

    Inputs:
        data: nxm numpy array (n=num signals, m = num dimensions)
        window_type ('hann'): string ('boxcar', 'gaussian', 'hann', 'bartlett', 'blackman')
        filter_width (11): int (wider is more smooth) odd is ideal
        sigma (2.): scalar std deviation only used for gaussian window
    Outputs
        data_smoothed: nxm array after being smoothed (nxm 2d array)
    """
    nrows, ncols = data.shape
    data_smoothed = []
    for row in range(nrows):
        smoothed_row, _ = smooth(data[row, :],
                                 window_type=window_type,
                                 filter_width=filter_width,
                                 sigma=sigma,
                                 plot_on=False)
        data_smoothed.append(smoothed_row)
    return np.asarray(data_smoothed)


def power_spec(data,
               sampling_frequency,
               segment_length=1024,
               segment_overlap=512,
               window='hann',
               view_range=None,
               scaling='density',
               plot_on=0):
    """
    Calculates power spectral density (default), or power spectrum, of signal
    using Welch's method.
    Wrapper for scipy.signal.welch

    Inputs:
        data: numpy array
        sampling_frequency (float): frequency (Hz) at which signals were sampled
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512) (must be less than segment_length)
        window (string): type of window to apply to each segment to make it periodic ('hann')
        view_range (2-elt array-like): low and high frequencies used only for plotting (None)
        scaling (str): 'density' for power spectral density, 'spectrum' for power spectrum ('density')
        plot_on (int): 0 for no plotting, 1 to plot spectrum (0)

    Outputs:
        ps: the power spectral density, or power spectrum of the signal
        frequencies: frequencies of spectrum

    Note:
        Frequencies range from sampling_frequency/num_points up to sampling_frequency/2,
        in increments of sampling_frequency/num_points.
    """

    frequencies, ps = welch(data.reshape(-1,),
                            fs=sampling_frequency,  # sample rate
                            window='hann',
                            nperseg=segment_length,
                            noverlap=segment_overlap,
                            detrend='constant',
                            average='mean',  # other option is 'median'
                            scaling=scaling)

    if plot_on:
        first_ind, last_ind = ind_limits(frequencies, view_range)
        plt.figure()
        plt.semilogy(frequencies[first_ind: last_ind],
                     ps[first_ind: last_ind],
                     color=(0.4, 0.4, 0.4),
                     linewidth=0.75)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel('Frequency')
        plt.ylabel('log(Power)')

    return ps, frequencies


def bandpass_filter(data, lowcut, highcut, sampling_frequency, order=5, plot_on=0):
    """
    Bandpass filter signal between locut and highcut frequencies
    Uses butterworth second order section.
    Wrapper for scipy.signal.butter
    Adapted from: https://stackoverflow.com/a/48677312/1886357

    Inputs:
        data (1d numpy array)
        lowcut: low cutoff frequency
        highcut: high cutoff frequency
        sampling_frequency: frequency (Hz) at which data was sampled
        order: order of filter (higher is sharper corners) (5)
        plot_on (int): 0 no plot, 1 to plot filter, original, and filtered signals

    Outputs:
        data_filtered (1d numpy array) -- same size as data, but filtered
        butter_sos: butterworth bandpass filter (second order section)

    """
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    butter_sos = signal.butter(
        order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = signal.sosfiltfilt(butter_sos, data)
    if plot_on:
        print("Plotting")
        #  Filter
        w, h = signal.sosfreqz(butter_sos, worN=2000)
        plt.subplot(2, 1, 1)
        plt.plot((sampling_frequency * 0.5 / np.pi) * w, abs(h))
        plt.axvline(lowcut, color='r', linewidth=0.5)
        plt.axvline(highcut, color='r', linewidth=0.5)
        plt.ylabel('Gain')
        plt.xlabel('Frequency')
        plt.autoscale(enable=True, axis='x', tight=True)

        # Data: original and filtered
        plt.subplot(2, 1, 2)
        plt.plot(data, color=(0.7, 0.7, 0.7), linewidth=0.5)
        plt.plot(filtered_data, color='r', linewidth=1)
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.autoscale(enable=True, axis='x', tight=True)

        plt.tight_layout()

    return filtered_data, butter_sos


def notch_filter(data, notch_frequency, sampling_frequency, quality_factor=35., plot_on=1):
    """
    Apply a notch filter at notch_frequency to 1d data (can remove 60Hz for instance)
    Wrapper for scipy.signal.iirnotch

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
    data = data.flatten()

    b, a = signal.iirnotch(notch_frequency, quality_factor, sampling_frequency)
    data_filtered = signal.filtfilt(b, a, data)

    if plot_on:
        # Frequency response
        freq, h = signal.freqz(b, a, fs=sampling_frequency)
        ig, axs = plt.subplots(3, 1, figsize=(12, 10))
        # Filter
        axs[0].plot(freq, 20*np.log10(abs(h)))
        axs[0].autoscale(enable=True, axis='x', tight=True)
        axs[0].set_xlabel('Frequency')
        axs[0].set_ylabel('Power')
        # Original signal and filtered version of signal
        axs[1].plot(data, color=(0.2, 0.2, 0.2), linewidth=1)
        axs[1].autoscale(enable=True, axis='x', tight=True)
        axs[2].plot(data_filtered,
                    color=(0.2, 0.2, 0.2),
                    linewidth=1)
        axs[2].autoscale(enable=True, axis='x', tight=True)
        axs[2].set_xlabel('Sample')
        axs[2].get_shared_x_axes().join(axs[1], axs[2])
        axs[2].get_shared_y_axes().join(axs[1], axs[2])
        plt.suptitle('Notch filtered Data', y=1)
        plt.tight_layout()

    return data_filtered, b, a


def spectrogram(data,
                sampling_frequency,
                segment_length=1024,
                segment_overlap=512,
                window='hann',
                view_range=None,
                all_events=None,
                colormap='inferno',
                notch_frequency=None,
                plot_on=0,
                scaling='density'):
    """
    Get/plot spectrogram of signal
    Wrapper for scipy.spectrogram

    Inputs:
        data: numpy array
        sampling_frequency (float): frequency (Hz) at which samples were acquired
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512) (must be less than segment_length)
        window (string): type of window to apply to each segment to make it periodic
        view_range (2-elt array-like): low and high frequencies used only for plotting (None)
        all_events (list of lists): times to show vertical bands for events, used for plotting
        colormap (string): colormap (inferno) (see also gist_heat, twilight_shifted, jet, ocean, bone)
        notch_frequency (float): if you want to filter out a frequency first (None)
        plot_on (int): 0 for no plotting, 1 to plot signal/spectrogram (0)
        scaling (str): 'density' for power spectral density, 'spectrum' for power spectrum ('density')

    Outputs:
        spectrogram (num_freqs x num_time_points) power spectral density over time
        freqs (array of frequencies): from `sampling_frequency/segment_length` up to `sampling_frequency/2`
        time_bins (time bin centers): can control resolution w/segment_overlap. DeltaT = (segment_length-segment_overlap)/sampling_freq
        axs: axes (None if plot_on is 0)

    Notes:
        - To plot use pcolormesh and 10*log10(spectrogram) otherwise it will look weird.
        - Windowing is not for smoothing, but to extract the data for the short-time FFT --
           the segment_length window makes the data segment quasi-periodic (wraps around
           values as the window drops to zero). This makes the FFT behave. Do not use
           boxcar I would stick with hann or similar.
    To do
        - I had it using different colors for different events but removed this. Maybe add
        this feature back at some point.
    """
    if data.ndim > 1:
        data = data.flatten()

    if notch_frequency is not None:
        data, _, _ = notch_filter(data,
                                  notch_frequency,
                                  sampling_frequency,
                                  plot_on=0)

    freqs, time_bins, spect = signal.spectrogram(data,
                                                 fs=sampling_frequency,
                                                 nperseg=segment_length,
                                                 noverlap=segment_overlap,
                                                 window=window,
                                                 detrend='constant',  # removes mean from each segment
                                                 scaling=scaling)
    if plot_on:
        num_samples = len(data)
        sampling_period = 1/sampling_frequency
        duration = num_samples*sampling_period
        times = np.linspace(0, duration, num_samples)
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        # Plot raw signal
        axs[0].plot(times, data, color=(0.5, 0.5, 0.5), linewidth=0.5)
        axs[0].autoscale(enable=True, axis='x', tight=True)
        # Plot spectrogram
        first_ind, last_ind = ind_limits(freqs, view_range)
        axs[1].pcolormesh(time_bins,
                          freqs[first_ind:last_ind],
                          10*np.log10(spect[first_ind: last_ind, :]), cmap=colormap)
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlabel('t(s)')
        axs[1].autoscale(enable=True, axis='x', tight=True)
        # Plot events
        if all_events is not None:
            for event_ind, events in enumerate(all_events):
                for event in events:
                    axs[0].axvline(x=event, zorder=3, color='k', linewidth=0.5)
                    axs[1].axvline(x=event, zorder=3, color='k', linewidth=0.5)
        plt.tight_layout()
    else:
        axs = None

    return spect, freqs, time_bins, axs


# %%  run some tests
if __name__ == '__main__':
    plt.close('all')
    """
    Test smooth
    """
    print("anaties.signals: testing smooth()...")
    std = 0.4
    t = np.linspace(-1, 1, 201)
    pure_signal = (np.sin(2 * np.pi * 0.75 * t*(1-t) + 2.1) +
                   0.1*np.sin(2 * np.pi * 1.25 * t + 1) +
                   0.18*np.cos(2 * np.pi * 3.85 * t))
    noisy_signal = pure_signal + \
        np.random.normal(loc=0, scale=std, size=t.shape)
    filter_width = 13
    window = 'gaussian'
    smoothed_signal, gauss_window = smooth(noisy_signal,
                                           window_type=window,
                                           filter_width=13,
                                           sigma=3,
                                           plot_on=1)
    plt.title(f'signals.smooth test with {window} filter')
    plt.show()

    """
    Test smooth_rows()
    """
    print("anaties.signals: testing smooth_rows()")
    noisy_array = [pure_signal+np.random.normal(scale=std, size=t.shape),
                   pure_signal+2+np.random.normal(scale=std, size=t.shape),
                   pure_signal-2+np.random.normal(scale=std, size=t.shape)]
    noisy_array = np.asarray(noisy_array)
    smoothed_array = smooth_rows(noisy_array,
                                 window_type='gaussian',
                                 sigma=3,
                                 filter_width=21)
    plt.figure("smooth_rows()")
    plt.plot(noisy_array.T, linewidth=0.5, color=(0.5, 0.5, 0.5))
    plt.plot(smoothed_array.T, linewidth=1, color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title('signals.smoothed_rows test')
    plt.show()

    """
    Test power_spec
    """
    print("\nanaties.signals: testing power_spec()...")
    f1 = 20
    f2 = 33
    num_points = 1000   # Number of points
    samp_pd = 0.001  # sampling period
    samp_freq = 1/samp_pd
    x = np.linspace(0.0, num_points*samp_pd, num_points)
    y = np.sin(f1 * 2.0*np.pi*x) + 0.5*np.sin(f2 * 2.0*np.pi*x)
    spectral_dens, freqs = power_spec(y,
                                      samp_freq,
                                      segment_length=512,
                                      segment_overlap=200,
                                      view_range=[3, 50],
                                      plot_on=1)
    plt.title('signals.power_spec() test')
    plt.axvline(f1, color='r')
    plt.axvline(f2, color='r')
    plt.show()

    """
    Test spectrogram
    """
    # First extract some sample audio data to analyze
    wav_path = r'../data/songbirds.wav'
    sample_rate, data_full = wavfile.read(wav_path)
    start_ind = 3_450_000
    num_samples = 300_000  # 1_500_000
    data = data_full[start_ind: start_ind+num_samples, 0]
    segment_length = 1024
    segment_overlap = segment_length//2
    event1 = [1.86, 5.3]
    event2 = [2.86, 6.3]
    spect, spect_freqs, spect_time_bins, axs = spectrogram(data,
                                                           sample_rate,
                                                           segment_length=1024,
                                                           segment_overlap=512,
                                                           window='hann',
                                                           view_range=[
                                                               300, 15_000],
                                                           all_events=[
                                                               event1, event2],
                                                           plot_on=1)
    plt.suptitle('signals.spectrogram test', y=1)
    plt.show()

    """
    Test notch filter
    """
    print("\nanaties.signals: testing notch_filter()...")
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
                                       quality_factor=35.,
                                       plot_on=1)
    plt.suptitle('signals.notch filter test', y=1)
    plt.show()

    """
    test bandpass filter
    """
    print("\nanaties.signals: testing bandpass_filter()...")
    samp_freq = 1000  # Sample frequency (Hz)
    f1 = 13
    f2 = 27
    f3 = 60
    std = 0.4
    num_points = 3_000   # Number of points
    samp_freq = 2000
    samp_pd = 1/samp_freq  # sampling period
    duration = num_points * samp_pd
    t = np.linspace(0.0, duration, num_points)
    y_pure = np.sin(f1 * 2.0*np.pi*t) + np.sin(f2 * 2.0 *
                                               np.pi*t) + np.sin(f3 * 2.0*np.pi*t)
    y_noisy = y_pure + np.random.normal(loc=0, scale=std, size=y_pure.shape)
    low_cut = 20
    high_cut = 34
    filter_order = 5  # can mess with t his
    filtered_y, sos_filter = bandpass_filter(
        y_noisy, low_cut, high_cut, samp_freq, order=5, plot_on=1)

    print("\nanaties.signals: tests done...")
    # Tests done

# %%


def smooth(data, window_type='hann', filter_width=11, sigma=2, plot_on=1):
    """
    Smooth 1d data with moving window (uses filtfilt to have zero phase distortion)
    Wrapper for scipy.signal.filtfilt
    To do: consider replacing with sosfiltfilt

    Inputs:
        data: numpy array
        window_type ('hann'): string ('boxcar', 'gaussian', 'hann', 'bartlett', 'blackman')
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
                                    data, method="gust")  # pad

    if plot_on:
        if plot_on > 1:
            plt.plot(filter_window)
            plt.title(f'{window_type} filter')
        plt.figure('signal', figsize=(10, 5))
        plt.plot(data, color=(0.7, 0.7, 0.7),
                 label='noisy signal', linewidth=1)
        plt.plot(data_smoothed, color='r', label='smoothed signal')
        plt.xlim(0, len(data_smoothed))
        plt.xlabel('sample')
        plt.grid(True)
        plt.legend()

    return data_smoothed, filter_window


def smooth_rows(data, window_type='hann', filter_width=11, sigma=2):
    """
    Smooth each row of a 2d array: uses smooth() see that for more details
    for the params.

    Inputs:
        data: nxm numpy array (n=num signals, m = num dimensions)
        window_type ('hann'): string ('boxcar', 'gaussian', 'hann', 'bartlett', 'blackman')
        filter_width (11): int (wider is more smooth) odd is ideal
        sigma (2.): scalar std deviation only used for gaussian window
    Outputs
        data_smoothed: nxm array after being smoothed (nxm 2d array)
    """
    nrows, ncols = data.shape
    data_smoothed = []
    for row in range(nrows):
        smoothed_row, _ = smooth(data[row, :],
                                 window_type=window_type,
                                 filter_width=filter_width,
                                 sigma=sigma,
                                 plot_on=False)
        data_smoothed.append(smoothed_row)
    return np.asarray(data_smoothed)


def power_spec(data,
               sampling_frequency,
               segment_length=1024,
               segment_overlap=512,
               window='hann',
               view_range=None,
               scaling='density',
               plot_on=0):
    """
    Calculates power spectral density (default), or power spectrum, of signal
    using Welch's method.
    Wrapper for scipy.signal.welch

    Inputs:
        data: numpy array
        sampling_frequency (float): frequency (Hz) at which signals were sampled
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512) (must be less than segment_length)
        window (string): type of window to apply to each segment to make it periodic ('hann')
        view_range (2-elt array-like): low and high frequencies used only for plotting (None)
        scaling (str): 'density' for power spectral density, 'spectrum' for power spectrum ('density')
        plot_on (int): 0 for no plotting, 1 to plot spectrum (0)

    Outputs:
        ps: the power spectral density, or power spectrum of the signal
        frequencies: frequencies of spectrum

    Note:
        Frequencies range from sampling_frequency/num_points up to sampling_frequency/2,
        in increments of sampling_frequency/num_points.
    """

    frequencies, ps = welch(data.reshape(-1,),
                            fs=sampling_frequency,  # sample rate
                            window='hann',
                            nperseg=segment_length,
                            noverlap=segment_overlap,
                            detrend='constant',
                            average='mean',  # other option is 'median'
                            scaling=scaling)

    if plot_on:
        first_ind, last_ind = ind_limits(frequencies, view_range)
        plt.figure()
        plt.semilogy(frequencies[first_ind: last_ind],
                     ps[first_ind: last_ind],
                     color=(0.4, 0.4, 0.4),
                     linewidth=0.75)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel('Frequency')
        plt.ylabel('log(Power)')

    return ps, frequencies


def bandpass_filter(data, lowcut, highcut, sampling_frequency, order=5, plot_on=0):
    """
    Bandpass filter signal between locut and highcut frequencies
    Uses butterworth second order section.
    Wrapper for scipy.signal.butter
    Adapted from: https://stackoverflow.com/a/48677312/1886357

    Inputs:
        data (1d numpy array)
        lowcut: low cutoff frequency
        highcut: high cutoff frequency
        sampling_frequency: frequency (Hz) at which data was sampled
        order: order of filter (higher is sharper corners) (5)
        plot_on (int): 0 no plot, 1 to plot filter, original, and filtered signals

    Outputs:
        data_filtered (1d numpy array) -- same size as data, but filtered
        butter_sos: butterworth bandpass filter (second order section)

    """
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    butter_sos = signal.butter(
        order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = signal.sosfiltfilt(butter_sos, data)
    if plot_on:
        print("Plotting")
        #  Filter
        w, h = signal.sosfreqz(butter_sos, worN=2000)
        plt.subplot(2, 1, 1)
        plt.plot((sampling_frequency * 0.5 / np.pi) * w, abs(h))
        plt.axvline(lowcut, color='r', linewidth=0.5)
        plt.axvline(highcut, color='r', linewidth=0.5)
        plt.ylabel('Gain')
        plt.xlabel('Frequency')
        plt.autoscale(enable=True, axis='x', tight=True)

        # Data: original and filtered
        plt.subplot(2, 1, 2)
        plt.plot(data, color=(0.7, 0.7, 0.7), linewidth=0.5)
        plt.plot(filtered_data, color='r', linewidth=1)
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.autoscale(enable=True, axis='x', tight=True)

        plt.tight_layout()

    return filtered_data, butter_sos


def notch_filter(data, notch_frequency, sampling_frequency, quality_factor=35., plot_on=1):
    """
    Apply a notch filter at notch_frequency to 1d data (can remove 60Hz for instance)
    Wrapper for scipy.signal.iirnotch

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
    data = data.flatten()

    b, a = signal.iirnotch(notch_frequency, quality_factor, sampling_frequency)
    data_filtered = signal.filtfilt(b, a, data)

    if plot_on:
        # Frequency response
        freq, h = signal.freqz(b, a, fs=sampling_frequency)
        ig, axs = plt.subplots(3, 1, figsize=(12, 10))
        # Filter
        axs[0].plot(freq, 20*np.log10(abs(h)))
        axs[0].autoscale(enable=True, axis='x', tight=True)
        axs[0].set_xlabel('Frequency')
        axs[0].set_ylabel('Power')
        # Original signal and filtered version of signal
        axs[1].plot(data, color=(0.2, 0.2, 0.2), linewidth=1)
        axs[1].autoscale(enable=True, axis='x', tight=True)
        axs[2].plot(data_filtered,
                    color=(0.2, 0.2, 0.2),
                    linewidth=1)
        axs[2].autoscale(enable=True, axis='x', tight=True)
        axs[2].set_xlabel('Sample')
        axs[2].get_shared_x_axes().join(axs[1], axs[2])
        axs[2].get_shared_y_axes().join(axs[1], axs[2])
        plt.suptitle('Notch filtered Data', y=1)
        plt.tight_layout()

    return data_filtered, b, a


def spectrogram(data,
                sampling_frequency,
                segment_length=1024,
                segment_overlap=512,
                window='hann',
                view_range=None,
                all_events=None,
                colormap='inferno',
                notch_frequency=None,
                plot_on=0,
                scaling='density'):
    """
    Get/plot spectrogram of signal
    Wrapper for scipy.spectrogram

    Inputs:
        data: numpy array
        sampling_frequency (float): frequency (Hz) at which samples were acquired
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512) (must be less than segment_length)
        window (string): type of window to apply to each segment to make it periodic
        view_range (2-elt array-like): low and high frequencies used only for plotting (None)
        all_events (list of lists): times to show vertical bands for events, used for plotting
        colormap (string): colormap (inferno) (see also gist_heat, twilight_shifted, jet, ocean, bone)
        notch_frequency (float): if you want to filter out a frequency first (None)
        plot_on (int): 0 for no plotting, 1 to plot signal/spectrogram (0)
        scaling (str): 'density' for power spectral density, 'spectrum' for power spectrum ('density')

    Outputs:
        spectrogram (num_freqs x num_time_points) power spectral density over time
        freqs (array of frequencies): from `sampling_frequency/segment_length` up to `sampling_frequency/2`
        time_bins (time bin centers): can control resolution w/segment_overlap. DeltaT = (segment_length-segment_overlap)/sampling_freq
        axs: axes (None if plot_on is 0)

    Notes:
        - To plot use pcolormesh and 10*log10(spectrogram) otherwise it will look weird.
        - Windowing is not for smoothing, but to extract the data for the short-time FFT --
           the segment_length window makes the data segment quasi-periodic (wraps around
           values as the window drops to zero). This makes the FFT behave. Do not use
           boxcar I would stick with hann or similar.
    To do
        - I had it using different colors for different events but removed this. Maybe add
        this feature back at some point.
    """
    if data.ndim > 1:
        data = data.flatten()

    if notch_frequency is not None:
        data, _, _ = notch_filter(data,
                                  notch_frequency,
                                  sampling_frequency,
                                  plot_on=0)

    freqs, time_bins, spect = signal.spectrogram(data,
                                                 fs=sampling_frequency,
                                                 nperseg=segment_length,
                                                 noverlap=segment_overlap,
                                                 window=window,
                                                 detrend='constant',  # removes mean from each segment
                                                 scaling=scaling)
    if plot_on:
        num_samples = len(data)
        sampling_period = 1/sampling_frequency
        duration = num_samples*sampling_period
        times = np.linspace(0, duration, num_samples)
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        # Plot raw signal
        axs[0].plot(times, data, color=(0.5, 0.5, 0.5), linewidth=0.5)
        axs[0].autoscale(enable=True, axis='x', tight=True)
        # Plot spectrogram
        first_ind, last_ind = ind_limits(freqs, view_range)
        axs[1].pcolormesh(time_bins,
                          freqs[first_ind:last_ind],
                          10*np.log10(spect[first_ind: last_ind, :]), cmap=colormap)
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlabel('t(s)')
        axs[1].autoscale(enable=True, axis='x', tight=True)
        # Plot events
        if all_events is not None:
            for event_ind, events in enumerate(all_events):
                for event in events:
                    axs[0].axvline(x=event, zorder=3, color='k', linewidth=0.5)
                    axs[1].axvline(x=event, zorder=3, color='k', linewidth=0.5)
        plt.tight_layout()
    else:
        axs = None

    return spect, freqs, time_bins, axs


# %%  run some tests
if __name__ == '__main__':
    plt.close('all')
    """
    Test smooth
    """
    print("anaties.signals: testing smooth()...")
    std = 0.4
    t = np.linspace(-1, 1, 201)
    pure_signal = (np.sin(2 * np.pi * 0.75 * t*(1-t) + 2.1) +
                   0.1*np.sin(2 * np.pi * 1.25 * t + 1) +
                   0.18*np.cos(2 * np.pi * 3.85 * t))
    noisy_signal = pure_signal + \
        np.random.normal(loc=0, scale=std, size=t.shape)
    filter_width = 13
    window = 'gaussian'
    smoothed_signal, gauss_window = smooth(noisy_signal,
                                           window_type=window,
                                           filter_width=13,
                                           sigma=3,
                                           plot_on=1)
    plt.title(f'signals.smooth test with {window} filter')
    plt.show()

    """
    Test smooth_rows()
    """
    print("anaties.signals: testing smooth_rows()")
    noisy_array = [pure_signal+np.random.normal(scale=std, size=t.shape),
                   pure_signal+2+np.random.normal(scale=std, size=t.shape),
                   pure_signal-2+np.random.normal(scale=std, size=t.shape)]
    noisy_array = np.asarray(noisy_array)
    smoothed_array = smooth_rows(noisy_array,
                                 window_type='gaussian',
                                 sigma=3,
                                 filter_width=21)
    plt.figure("smooth_rows()")
    plt.plot(noisy_array.T, linewidth=0.5, color=(0.5, 0.5, 0.5))
    plt.plot(smoothed_array.T, linewidth=1, color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title('signals.smoothed_rows test')
    plt.show()

    """
    Test power_spec
    """
    print("\nanaties.signals: testing power_spec()...")
    f1 = 20
    f2 = 33
    num_points = 1000   # Number of points
    samp_pd = 0.001  # sampling period
    samp_freq = 1/samp_pd
    x = np.linspace(0.0, num_points*samp_pd, num_points)
    y = np.sin(f1 * 2.0*np.pi*x) + 0.5*np.sin(f2 * 2.0*np.pi*x)
    spectral_dens, freqs = power_spec(y,
                                      samp_freq,
                                      segment_length=512,
                                      segment_overlap=200,
                                      view_range=[3, 50],
                                      plot_on=1)
    plt.title('signals.power_spec() test')
    plt.axvline(f1, color='r')
    plt.axvline(f2, color='r')
    plt.show()

    """
    Test spectrogram
    """
    # First extract some sample audio data to analyze
    wav_path = r'../data/songbirds.wav'
    sample_rate, data_full = wavfile.read(wav_path)
    start_ind = 3_450_000
    num_samples = 300_000  # 1_500_000
    data = data_full[start_ind: start_ind+num_samples, 0]
    segment_length = 1024
    segment_overlap = segment_length//2
    event1 = [1.86, 5.3]
    event2 = [2.86, 6.3]
    spect, spect_freqs, spect_time_bins, axs = spectrogram(data,
                                                           sample_rate,
                                                           segment_length=1024,
                                                           segment_overlap=512,
                                                           window='hann',
                                                           view_range=[
                                                               300, 15_000],
                                                           all_events=[
                                                               event1, event2],
                                                           plot_on=1)
    plt.suptitle('signals.spectrogram test', y=1)
    plt.show()

    """
    Test notch filter
    """
    print("\nanaties.signals: testing notch_filter()...")
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
                                       quality_factor=35.,
                                       plot_on=1)
    plt.suptitle('signals.notch filter test', y=1)
    plt.show()

    """
    test bandpass filter
    """
    print("\nanaties.signals: testing bandpass_filter()...")
    samp_freq = 1000  # Sample frequency (Hz)
    f1 = 13
    f2 = 27
    f3 = 60
    std = 0.4
    num_points = 3_000   # Number of points
    samp_freq = 2000
    samp_pd = 1/samp_freq  # sampling period
    duration = num_points * samp_pd
    t = np.linspace(0.0, duration, num_points)
    y_pure = np.sin(f1 * 2.0*np.pi*t) + np.sin(f2 * 2.0 *
                                               np.pi*t) + np.sin(f3 * 2.0*np.pi*t)
    y_noisy = y_pure + np.random.normal(loc=0, scale=std, size=y_pure.shape)
    low_cut = 20
    high_cut = 34
    filter_order = 5  # can mess with t his
    filtered_y, sos_filter = bandpass_filter(
        y_noisy, low_cut, high_cut, samp_freq, order=5, plot_on=1)

    print("\nanaties.signals: tests done...")
    # Tests done
"""
signal processing functions for anaties package

https://github.com/EricThomson/anaties
"""

# import scipy.fftpack as fftpack #not used

# sys.path.append(str(Path('.').absolute().parent))
# from anaties.helpers import ind_limits

# %%


def smooth(data, window_type='hann', filter_width=11, sigma=2, plot_on=1):
    """
    Smooth 1d data with moving window (uses filtfilt to have zero phase distortion)
    Wrapper for scipy.signal.filtfilt
    To do: consider replacing with sosfiltfilt

    Inputs:
        data: numpy array
        window_type ('hann'): string ('boxcar', 'gaussian', 'hann', 'bartlett', 'blackman')
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
                                    data, method="gust")  # pad

    if plot_on:
        if plot_on > 1:
            plt.plot(filter_window)
            plt.title(f'{window_type} filter')
        plt.figure('signal', figsize=(10, 5))
        plt.plot(data, color=(0.7, 0.7, 0.7),
                 label='noisy signal', linewidth=1)
        plt.plot(data_smoothed, color='r', label='smoothed signal')
        plt.xlim(0, len(data_smoothed))
        plt.xlabel('sample')
        plt.grid(True)
        plt.legend()

    return data_smoothed, filter_window


def smooth_rows(data, window_type='hann', filter_width=11, sigma=2):
    """
    Smooth each row of a 2d array: uses smooth() see that for more details
    for the params.

    Inputs:
        data: nxm numpy array (n=num signals, m = num dimensions)
        window_type ('hann'): string ('boxcar', 'gaussian', 'hann', 'bartlett', 'blackman')
        filter_width (11): int (wider is more smooth) odd is ideal
        sigma (2.): scalar std deviation only used for gaussian window
    Outputs
        data_smoothed: nxm array after being smoothed (nxm 2d array)
    """
    nrows, ncols = data.shape
    data_smoothed = []
    for row in range(nrows):
        smoothed_row, _ = smooth(data[row, :],
                                 window_type=window_type,
                                 filter_width=filter_width,
                                 sigma=sigma,
                                 plot_on=False)
        data_smoothed.append(smoothed_row)
    return np.asarray(data_smoothed)


def power_spec(data,
               sampling_frequency,
               segment_length=1024,
               segment_overlap=512,
               window='hann',
               view_range=None,
               scaling='density',
               plot_on=0):
    """
    Calculates power spectral density (default), or power spectrum, of signal
    using Welch's method.
    Wrapper for scipy.signal.welch

    Inputs:
        data: numpy array
        sampling_frequency (float): frequency (Hz) at which signals were sampled
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512) (must be less than segment_length)
        window (string): type of window to apply to each segment to make it periodic ('hann')
        view_range (2-elt array-like): low and high frequencies used only for plotting (None)
        scaling (str): 'density' for power spectral density, 'spectrum' for power spectrum ('density')
        plot_on (int): 0 for no plotting, 1 to plot spectrum (0)

    Outputs:
        ps: the power spectral density, or power spectrum of the signal
        frequencies: frequencies of spectrum

    Note:
        Frequencies range from sampling_frequency/num_points up to sampling_frequency/2,
        in increments of sampling_frequency/num_points.
    """

    frequencies, ps = welch(data.reshape(-1,),
                            fs=sampling_frequency,  # sample rate
                            window='hann',
                            nperseg=segment_length,
                            noverlap=segment_overlap,
                            detrend='constant',
                            average='mean',  # other option is 'median'
                            scaling=scaling)

    if plot_on:
        first_ind, last_ind = ind_limits(frequencies, view_range)
        plt.figure()
        plt.semilogy(frequencies[first_ind: last_ind],
                     ps[first_ind: last_ind],
                     color=(0.4, 0.4, 0.4),
                     linewidth=0.75)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel('Frequency')
        plt.ylabel('log(Power)')

    return ps, frequencies


def bandpass_filter(data, lowcut, highcut, sampling_frequency, order=5, plot_on=0):
    """
    Bandpass filter signal between locut and highcut frequencies
    Uses butterworth second order section.
    Wrapper for scipy.signal.butter
    Adapted from: https://stackoverflow.com/a/48677312/1886357

    Inputs:
        data (1d numpy array)
        lowcut: low cutoff frequency
        highcut: high cutoff frequency
        sampling_frequency: frequency (Hz) at which data was sampled
        order: order of filter (higher is sharper corners) (5)
        plot_on (int): 0 no plot, 1 to plot filter, original, and filtered signals

    Outputs:
        data_filtered (1d numpy array) -- same size as data, but filtered
        butter_sos: butterworth bandpass filter (second order section)

    """
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    butter_sos = signal.butter(
        order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = signal.sosfiltfilt(butter_sos, data)
    if plot_on:
        print("Plotting")
        #  Filter
        w, h = signal.sosfreqz(butter_sos, worN=2000)
        plt.subplot(2, 1, 1)
        plt.plot((sampling_frequency * 0.5 / np.pi) * w, abs(h))
        plt.axvline(lowcut, color='r', linewidth=0.5)
        plt.axvline(highcut, color='r', linewidth=0.5)
        plt.ylabel('Gain')
        plt.xlabel('Frequency')
        plt.autoscale(enable=True, axis='x', tight=True)

        # Data: original and filtered
        plt.subplot(2, 1, 2)
        plt.plot(data, color=(0.7, 0.7, 0.7), linewidth=0.5)
        plt.plot(filtered_data, color='r', linewidth=1)
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.autoscale(enable=True, axis='x', tight=True)

        plt.tight_layout()

    return filtered_data, butter_sos


def notch_filter(data, notch_frequency, sampling_frequency, quality_factor=35., plot_on=1):
    """
    Apply a notch filter at notch_frequency to 1d data (can remove 60Hz for instance)
    Wrapper for scipy.signal.iirnotch

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
    data = data.flatten()

    b, a = signal.iirnotch(notch_frequency, quality_factor, sampling_frequency)
    data_filtered = signal.filtfilt(b, a, data)

    if plot_on:
        # Frequency response
        freq, h = signal.freqz(b, a, fs=sampling_frequency)
        ig, axs = plt.subplots(3, 1, figsize=(12, 10))
        # Filter
        axs[0].plot(freq, 20*np.log10(abs(h)))
        axs[0].autoscale(enable=True, axis='x', tight=True)
        axs[0].set_xlabel('Frequency')
        axs[0].set_ylabel('Power')
        # Original signal and filtered version of signal
        axs[1].plot(data, color=(0.2, 0.2, 0.2), linewidth=1)
        axs[1].autoscale(enable=True, axis='x', tight=True)
        axs[2].plot(data_filtered,
                    color=(0.2, 0.2, 0.2),
                    linewidth=1)
        axs[2].autoscale(enable=True, axis='x', tight=True)
        axs[2].set_xlabel('Sample')
        axs[2].get_shared_x_axes().join(axs[1], axs[2])
        axs[2].get_shared_y_axes().join(axs[1], axs[2])
        plt.suptitle('Notch filtered Data', y=1)
        plt.tight_layout()

    return data_filtered, b, a


def spectrogram(data,
                sampling_frequency,
                segment_length=1024,
                segment_overlap=512,
                window='hann',
                view_range=None,
                all_events=None,
                colormap='inferno',
                notch_frequency=None,
                plot_on=0,
                scaling='density'):
    """
    Get/plot spectrogram of signal
    Wrapper for scipy.spectrogram

    Inputs:
        data: numpy array
        sampling_frequency (float): frequency (Hz) at which samples were acquired
        segment_length (int): number of samples per segment in which to calculate STFFT (1024)
        segment_overlap (int): overlap samples between segments (512) (must be less than segment_length)
        window (string): type of window to apply to each segment to make it periodic
        view_range (2-elt array-like): low and high frequencies used only for plotting (None)
        all_events (list of lists): times to show vertical bands for events, used for plotting
        colormap (string): colormap (inferno) (see also gist_heat, twilight_shifted, jet, ocean, bone)
        notch_frequency (float): if you want to filter out a frequency first (None)
        plot_on (int): 0 for no plotting, 1 to plot signal/spectrogram (0)
        scaling (str): 'density' for power spectral density, 'spectrum' for power spectrum ('density')

    Outputs:
        spectrogram (num_freqs x num_time_points) power spectral density over time
        freqs (array of frequencies): from `sampling_frequency/segment_length` up to `sampling_frequency/2`
        time_bins (time bin centers): can control resolution w/segment_overlap. DeltaT = (segment_length-segment_overlap)/sampling_freq
        axs: axes (None if plot_on is 0)

    Notes:
        - To plot use pcolormesh and 10*log10(spectrogram) otherwise it will look weird.
        - Windowing is not for smoothing, but to extract the data for the short-time FFT --
           the segment_length window makes the data segment quasi-periodic (wraps around
           values as the window drops to zero). This makes the FFT behave. Do not use
           boxcar I would stick with hann or similar.
    To do
        - I had it using different colors for different events but removed this. Maybe add
        this feature back at some point.
    """
    if data.ndim > 1:
        data = data.flatten()

    if notch_frequency is not None:
        data, _, _ = notch_filter(data,
                                  notch_frequency,
                                  sampling_frequency,
                                  plot_on=0)

    freqs, time_bins, spect = signal.spectrogram(data,
                                                 fs=sampling_frequency,
                                                 nperseg=segment_length,
                                                 noverlap=segment_overlap,
                                                 window=window,
                                                 detrend='constant',  # removes mean from each segment
                                                 scaling=scaling)
    if plot_on:
        num_samples = len(data)
        sampling_period = 1/sampling_frequency
        duration = num_samples*sampling_period
        times = np.linspace(0, duration, num_samples)
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        # Plot raw signal
        axs[0].plot(times, data, color=(0.5, 0.5, 0.5), linewidth=0.5)
        axs[0].autoscale(enable=True, axis='x', tight=True)
        # Plot spectrogram
        first_ind, last_ind = ind_limits(freqs, view_range)
        axs[1].pcolormesh(time_bins,
                          freqs[first_ind:last_ind],
                          10*np.log10(spect[first_ind: last_ind, :]), cmap=colormap)
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlabel('t(s)')
        axs[1].autoscale(enable=True, axis='x', tight=True)
        # Plot events
        if all_events is not None:
            for event_ind, events in enumerate(all_events):
                for event in events:
                    axs[0].axvline(x=event, zorder=3, color='k', linewidth=0.5)
                    axs[1].axvline(x=event, zorder=3, color='k', linewidth=0.5)
        plt.tight_layout()
    else:
        axs = None

    return spect, freqs, time_bins, axs


# %%  run some tests
if __name__ == '__main__':
    plt.close('all')
    """
    Test smooth
    """
    print("anaties.signals: testing smooth()...")
    std = 0.4
    t = np.linspace(-1, 1, 201)
    pure_signal = (np.sin(2 * np.pi * 0.75 * t*(1-t) + 2.1) +
                   0.1*np.sin(2 * np.pi * 1.25 * t + 1) +
                   0.18*np.cos(2 * np.pi * 3.85 * t))
    noisy_signal = pure_signal + \
        np.random.normal(loc=0, scale=std, size=t.shape)
    filter_width = 13
    window = 'gaussian'
    smoothed_signal, gauss_window = smooth(noisy_signal,
                                           window_type=window,
                                           filter_width=13,
                                           sigma=3,
                                           plot_on=1)
    plt.title(f'signals.smooth test with {window} filter')
    plt.show()

    """
    Test smooth_rows()
    """
    print("anaties.signals: testing smooth_rows()")
    noisy_array = [pure_signal+np.random.normal(scale=std, size=t.shape),
                   pure_signal+2+np.random.normal(scale=std, size=t.shape),
                   pure_signal-2+np.random.normal(scale=std, size=t.shape)]
    noisy_array = np.asarray(noisy_array)
    smoothed_array = smooth_rows(noisy_array,
                                 window_type='gaussian',
                                 sigma=3,
                                 filter_width=21)
    plt.figure("smooth_rows()")
    plt.plot(noisy_array.T, linewidth=0.5, color=(0.5, 0.5, 0.5))
    plt.plot(smoothed_array.T, linewidth=1, color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title('signals.smoothed_rows test')
    plt.show()

    """
    Test power_spec
    """
    print("\nanaties.signals: testing power_spec()...")
    f1 = 20
    f2 = 33
    num_points = 1000   # Number of points
    samp_pd = 0.001  # sampling period
    samp_freq = 1/samp_pd
    x = np.linspace(0.0, num_points*samp_pd, num_points)
    y = np.sin(f1 * 2.0*np.pi*x) + 0.5*np.sin(f2 * 2.0*np.pi*x)
    spectral_dens, freqs = power_spec(y,
                                      samp_freq,
                                      segment_length=512,
                                      segment_overlap=200,
                                      view_range=[3, 50],
                                      plot_on=1)
    plt.title('signals.power_spec() test')
    plt.axvline(f1, color='r')
    plt.axvline(f2, color='r')
    plt.show()

    """
    Test spectrogram
    """
    # First extract some sample audio data to analyze
    wav_path = r'../data/songbirds.wav'
    sample_rate, data_full = wavfile.read(wav_path)
    start_ind = 3_450_000
    num_samples = 300_000  # 1_500_000
    data = data_full[start_ind: start_ind+num_samples, 0]
    segment_length = 1024
    segment_overlap = segment_length//2
    event1 = [1.86, 5.3]
    event2 = [2.86, 6.3]
    spect, spect_freqs, spect_time_bins, axs = spectrogram(data,
                                                           sample_rate,
                                                           segment_length=1024,
                                                           segment_overlap=512,
                                                           window='hann',
                                                           view_range=[
                                                               300, 15_000],
                                                           all_events=[
                                                               event1, event2],
                                                           plot_on=1)
    plt.suptitle('signals.spectrogram test', y=1)
    plt.show()

    """
    Test notch filter
    """
    print("\nanaties.signals: testing notch_filter()...")
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
                                       quality_factor=35.,
                                       plot_on=1)
    plt.suptitle('signals.notch filter test', y=1)
    plt.show()

    """
    test bandpass filter
    """
    print("\nanaties.signals: testing bandpass_filter()...")
    samp_freq = 1000  # Sample frequency (Hz)
    f1 = 13
    f2 = 27
    f3 = 60
    std = 0.4
    num_points = 3_000   # Number of points
    samp_freq = 2000
    samp_pd = 1/samp_freq  # sampling period
    duration = num_points * samp_pd
    t = np.linspace(0.0, duration, num_points)
    y_pure = np.sin(f1 * 2.0*np.pi*t) + np.sin(f2 * 2.0 *
                                               np.pi*t) + np.sin(f3 * 2.0*np.pi*t)
    y_noisy = y_pure + np.random.normal(loc=0, scale=std, size=y_pure.shape)
    low_cut = 20
    high_cut = 34
    filter_order = 5  # can mess with t his
    filtered_y, sos_filter = bandpass_filter(
        y_noisy, low_cut, high_cut, samp_freq, order=5, plot_on=1)

    print("\nanaties.signals: tests done...")
    # Tests done
