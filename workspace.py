"""
For code under construction

Currently working on Spectrogram

Stuff I need to work out 
1. get scipy version working.
2, see how it relates to matplotlib specgram
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
   https://stackoverflow.com/questions/34156050/python-matplotlib-specgram-data-array-values-does-not-match-specgram-plot
   https://stackoverflow.com/questions/48598994/scipy-signal-spectrogram-compared-to-matplotlib-pyplot-specgram
   https://old.reddit.com/r/DSP/comments/bk1s3t/does_anybody_have_some_experience_using_scipy_and/#bottom-comments
    
specgram why?
    https://dsp.stackexchange.com/questions/1593/improving-spectrogram-resolution-in-python
    https://github.com/matplotlib/matplotlib/blob/d7feb03da5b78e15b002b7438779068a318a3024/lib/matplotlib/mlab.py

wavelet stuff -- this looks amazing:
    http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
    
    compare pywavelet to this:
    https://www.mathworks.com/help/wavelet/ref/cwt.html
    makre sure you get this:
    https://www.mathworks.com/help/wavelet/ug/boundary-effects-and-the-cone-of-influence.html
    
    Consider this:
    
        
    And why the fuck do they plot periods nad not just convert to damned freqs?
    
Wavelet vs spectrogram vs harmonics vs wow.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6709234/
    

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

#%% load full data and extract duration/time points
wav_path = r'./data/songbirds.wav'
sample_rate, data_full = wavfile.read(wav_path)
num_samples_full = data_full.shape[0]

duration_full = data_full.shape[0]/sample_rate
times_full = np.linspace(0, duration_full, data_full.shape[0])

print(f"Num samples total (millions): {num_samples_full/1e6}")
print(f"Time (seconds): {duration_full}")


#%% plot full data
plt.plot(data_full[:,0], color = 'r', label = "Left")
plt.plot(data_full[:,1], color = 'b', label = "Right")
plt.legend(loc = 'lower left')
plt.autoscale(enable=True, axis='x', tight=True)

#%% extract subset of full data
start_ind = 3_450_000
num_samples = 1_500_000
data = data_full[start_ind: start_ind+num_samples, 0]
duration = data.shape[0]/sample_rate
times = np.linspace(0, duration, data.shape[0])

print(f"Signal subset is {duration:0.2f}s starts {start_ind/sample_rate:0.2f}s in")

#%% Plot subset
plt.figure('bird signal', figsize=(12,5))
plt.plot(times, data, color = (0.5, 0.5, 0.5), linewidth = 0.5)
plt.xlabel('t(s)')
plt.autoscale(enable=True, axis='x', tight=True)

#%% matplotlib built-in (i mean...seriously?)
Pxx, freqs, bins, im =  plt.specgram(data, NFFT=1024, Fs = sample_rate, noverlap = 900)


#%%
freqs, times, spectrogram = signal.spectrogram(data, sample_rate)
plt.figure(figsize=(5, 4))
plt.pcolormesh(times, freqs, 10*np.log10(spectrogram))
#plt.imshow(np.log(spectrogram), aspect='auto', cmap='hot_r', origin='lower')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')

#%% wavelet
cwtmatr = signal.cwt(data, signal.ricker, widths)

plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',

           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())


#%% from scipy lecture notes
# https://raw.githubusercontent.com/scipy-lectures/scipy-lecture-notes/a9dc0e7ab0232041779509a0ae81b870be0fdbd2/intro/scipy/examples/plot_spectrogram.py
# Good test, but it looks like utter crap why would you use a chirp signal to demonstrate this?
# Generate a chirp signal
time_step = .01
time_vec = np.arange(0, 70, time_step)
# A signal with a small frequency chirp
sig = np.sin(0.5 * np.pi * time_vec * (1 + .1 * time_vec))
plt.figure(figsize=(8, 5))
plt.plot(time_vec, sig)
plt.autoscale(enable = True, axis = 'x', tight = True)

#  compute/plot spectrogram
freqs, times, spectrogram = signal.spectrogram(sig)
plt.figure(figsize=(5, 4))
plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')


#%%


#%%  Weird ass matplotlib demo
# OK this is just weird why is this even maintained like this?
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.0005
t = np.arange(0.0, 20.0, dt)
s1 = np.sin(2 * np.pi * 100 * t)
s2 = 2 * np.sin(2 * np.pi * 400 * t)

# create a transient "chirp"
s2[t <= 10] = s2[12 <= t] = 0

# add some noise into the mix
nse = 0.01 * np.random.random(size=len(t))

x = s1 + s2 + nse  # the signal
NFFT = 1024  # the length of the windowing segments
Fs = int(1.0 / dt)  # the sampling frequency

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot(t, x)
Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)

Pxx, freqs, bins, im = plt.specgram(data_full[:,0], NFFT=1024, Fs = sampfreq, noverlap = 900)
# The `specgram` method returns 4 objects. They are:
# - Pxx: the periodogram
# - freqs: the frequency vector
# - bins: the centers of the time bins
# - im: the matplotlib.image.AxesImage instance representing the data in the plot
plt.show()