# anaties
An analysis utilities package provides a thin wrapper for useful functions I use during analysis.


## Install
Eventually I will build a way to install but for now:

    conda create -n anaties
    conda activate anaties
    conda install python=3.7
    conda install scipy numpy matplotlib
    conda install -c conda-forge opencv=4

Optional stuff -- I install spyder. Eventually I might make notebooks in which case I'd install jupyter and nodejs.

What we have so far    

    signals.py
        - smooth: smooth a signal with a filter
        - fft: get fft and power spectrum of a signal




## Useful sources
### Smoothing
- https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html

### FFT
- https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/
- https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html#sphx-glr-intro-scipy-auto-examples-plot-fftpack-py


## Notes
###  Why no gaussian filter?
I will add it once I switch from numpy to scipy for the window functions.  

### Edge artifacts
Handling edge artifacts can be tricky: you can pad it (with different parameters), and use Gustafsson's method. I like Gustaffson's method so went with that as the default. At some point I might tinker with that: again that will be a half day to really get it right. Frankly the decisions you make about your edges shouldn't make much difference: if they do something has probably gone wrong with your design at a previous step.

### What about wavelets?
Right now for time-frequency analysis I'm going with spectrograms.  Wavelets are cool and there is a lot to be said for them in theory, but in practice with spectrograms there are a lot fewer ways to go off the rails. Wavelets require a subtle touch to get it right, and go against the plug-and-play spirit of this package. If you want to get started with wavelets in Python, I would recommend:   http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/


## To do
- Spectrograms
- Add numerical tests with random seed set not just graphical eyeball tests.

## Data sources
- Songbird wav from
 https://freesound.org/people/Sonic-ranger/sounds/243677/


  :)
