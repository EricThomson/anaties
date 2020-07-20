# anaties
An analysis utilities package. Mostly a thin wrapper for functions I find useful in scipy and other packages.


## Install
Eventually I will build a way to install but for now:

    conda create -n anaties
    conda activate anaties
    conda install python=3.7
    conda install scipy numpy matplotlib
    conda install -c conda-forge opencv=4

Optional stuff --
    pip install playsound # if you wan to play sounds with spectrograms
I install spyder. Eventually I might make notebooks in which case I'd install jupyter and nodejs.

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
Wavelets require a subtle touch to get it right, and go against the plug-and-play spirit of this package, so I've gone with spectrograms for now. I may add wavelets at some point, but if you want to get started with wavelets in Python, I recommend http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/

### Edge artifacts
Handling edge artifacts can be tricky: for now I am using Gustaffson's method  as the default, though at some point might tinker with that.

## To do
- Add numerical tests with random seed set not just graphical eyeball tests.
- Long-term: Make audio player that shows location in waveform for spectrogram.

## Data sources
Songbird wav is open source and available from https://freesound.org/people/Sonic-ranger/sounds/243677/

## About
Developed with the support of the Neurobehavioral Core at NIEHS: https://www.niehs.nih.gov/research/atniehs/facilities/neurobehavioral/index.cfm
