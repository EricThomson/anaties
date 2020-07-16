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


## To do
- Time-dependent frequency (spectrogram/wavelet)
- For smooth switch to scipy filter windows and add guassian:
https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows


  :)
