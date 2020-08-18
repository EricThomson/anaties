# anaties
An analysis utilities package. Mostly things I find useful for signal processing.


## Install
Eventually I will build a way to install but for now:

    conda create -n anaties
    conda activate anaties
    conda install python=3.7
    conda install scipy numpy matplotlib
    conda install -c conda-forge opencv=4

Install IDE if you want. Eventually I might make notebooks in which case I'd install jupyter and nodejs.

Brief summary of utilities:  

    signals.py (for 1d data arrays like LFP, sound, etc)
        - smooth: smooth a signal with a window (gaussian, etc)
        - fft: get fft and power spectrum of a signal
        - spectrogram: calculate/plot spectrogram of a signal
        - notch_filter: bandpass filter at specific frequency


    helpers.py (generic utility functions for use everywhere)
        - ind_limits: return indices that contain a range of data
        - rand_rgb: returns random array of rgb values useful for plotting

## Useful sources
### Smoothing
- https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html

### FFT
- https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/
- https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html#sphx-glr-intro-scipy-auto-examples-plot-fftpack-py


## Notes
### What about wavelets?
I may add wavelets at some point for time-frequency analysis, but if you want to get started with wavelets in Python, I recommend http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/

### Edge artifacts
Handling edge artifacts can be tricky: currently I use Gustaffson's method as the default, though at some point might tinker with that -- there are many options.


## To do
- ind_limits: add checks for data, data_limits, clarify description and docs
- Add playback of ephys signals (see notes in audio_playback_workspace), incorporate this into some tests of filtering, etc.. simpleaudio package is too simple I think.
- Add numerical tests with random seed set not just graphical eyeball tests.
- Long-term: autodocs (sphinx?)
- Long-term: Make audio player that shows location in waveform for spectrogram.

## Data sources
Songbird wav is open source and available from https://freesound.org/people/Sonic-ranger/sounds/243677/

## About
Developed with the support of the Neurobehavioral Core at NIEHS: https://www.niehs.nih.gov/research/atniehs/facilities/neurobehavioral/index.cfm
