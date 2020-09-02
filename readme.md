# anaties

An analysis utilities package. Common operations like signal smoothing that I find myself using in multiple projects.

Brief summary of utilities:  

    signals.py (for 1d data arrays like voltage, sound, etc)
        - smooth: smooth a signal with a window (gaussian, etc)
        - fft: get fft and power spectrum of a signal
        - spectrogram: calculate/plot spectrogram of a signal
        - notch_filter: bandpass filter at specific frequency


    helpers.py (generic utility functions for use everywhere)
        - datetime_string : return date_time string to use for naming files etc
        - ind_limits: return indices containing data within a range
        - rand_rgb: returns random array of rgb values useful for plotting

## Install
Eventually I will build a builder, but for now:

    conda create -n anaties
    conda activate anaties
    conda install python=3.7
    conda install scipy numpy matplotlib
    conda install -c conda-forge opencv=4

Voila. I import signals as signals, helpers as helpy.

## Useful sources
### Smoothing
- https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html

### FFT
- https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/
- https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html#sphx-glr-intro-scipy-auto-examples-plot-fftpack-py


## Notes
### Notes on FFT/spectrogram
- I subtract the mean of the signal/window before computing the power spectrum: otherwise the zero frequency power can spill over into lower frequencies.
- For the fft/spectrogram, the frequency components go from `samp_freq/num_points` up to `samp_freq/2`, in increments of `samp_freq/num_points`. To increase your frequency resolution, either increase your sampling rate, or number of points (`segment_length` for spectrogram).

### Edge artifacts
Handling edge artifacts can be tricky: currently I use Gustaffson's method as the default, though at some point might tinker with that -- there are many options.

### What about wavelets?
I may add wavelets at some point, but it isn't plug-and-play enough for this repo. If you want to get started with wavelets in Python, I recommend http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/



## Acknowledgments
- Songbird wav is open source from: https://freesound.org/people/Sonic-ranger/sounds/243677/
- Developed with the support of NIH Bioinformatics, and the Neurobehavioral Core at NIEHS.

## To do
- add ability to control event colors in spectrogram.
- ind_limits: add checks for data, data_limits, clarify description and docs
- Add audio playback of signals (see notes in audio_playback_workspace), incorporate this into some tests of filtering, etc.. simpleaudio package is too simple I think.
- Add numerical tests with random seed set not just graphical eyeball tests.
- Long-term: autodocs (sphinx?)
- Long-term: Make audio player that shows location in waveform for spectrogram.
