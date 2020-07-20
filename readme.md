# anaties
An analysis utilities package. Mostly a thin wrapper for functions I find useful in scipy and other packages.


## Install
Eventually I will build a way to install but for now:

    conda create -n anaties
    conda activate anaties
    conda install python=3.7
    conda install scipy numpy matplotlib
    conda install -c conda-forge opencv=4
    pip install simpleaudio

Install IDE if you want. Eventually I might make notebooks in which case I'd install jupyter and nodejs.

What we have so far    

    signals.py
        - smooth: smooth a signal with a filter
        - fft: get fft and power spectrum of a signal
        - spectrogram: calculate/plot spectrogram of a signal




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
Handling edge artifacts can be tricky: currently I use Gustaffson's method as the default, though at some point might tinker with that.

## To do
- Add ability to listen to filter (listen(data))
- Add notch filter to LFP? (notch(data, freq))
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
- Limit high frequencies in fft?
- Add numerical tests with random seed set not just graphical eyeball tests.
- Long-term: autodocs (sphinx?)
- Long-term: Make audio player that shows location in waveform for spectrogram.

## Data sources
Songbird wav is open source and available from https://freesound.org/people/Sonic-ranger/sounds/243677/

## About
Developed with the support of the Neurobehavioral Core at NIEHS: https://www.niehs.nih.gov/research/atniehs/facilities/neurobehavioral/index.cfm
