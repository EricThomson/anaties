# anaties
An analysis utilities package. Common operations like signal smoothing that I find myself using in multiple projects.

Brief summary of utilities:  

    signals.py (for 1d data arrays like voltage, sound, etc)
        - smooth: smooth a signal with a window (gaussian, etc)
        - smooth_rows: smooth each row of a 2d array using smooth()
        - fft: get fft and power spectrum of a signal
        - spectrogram: calculate/plot spectrogram of a signal
        - notch_filter: notch filter to attenuate specific frequency (e.g. 60hz)
        - bandpass_filter: allow through frequencies within low- and high-cutoff

    plots.py (basic plotting wrappers)
        - freqhist: calculate/plot a relative frequency histogram
        - paired_bar: bar plot for paired data
        - rect_highlight: overlay rectangular highlight on current figure

    stats (basic descriptive stats)
        - mn_sem: mean and std error of the mean of an array
        - mn_std: mean and standard deviation of an array
        - se_mean: std err of mean of array
        - se_median: std error of median of array
        - cramers_v: cramers v for effect size for chi-square test

    helpers.py (generic utility functions for use everywhere)
        - datetime_string : return date_time string to use for naming files etc
        - get_bins: get bin edges and centers, given limits and bin width
        - get_offdiag_vals: get lower off-diagonal values of a symmetric matrix
        - ind_limits: return indices that contain array data within range
        - is_symmetric: check if 2d array is symmetric
        - rand_rgb: returns random array of rgb values

## Install
Plan is to build an installer eventually. For now, using the anaconda prompt, just cd to the folder where you want the anaties folder placed, and:

    git clone https://github.com/EricThomson/anaties

This will download the package and place the anaties folder inside that folder. Then, to get it to work within any virtual environment, make sure you have the dependencies installed (scipy, numpy, matplotlib), and then you can import it with:

    sys.path.append(anaties_path)
    from anaties import signals as sig
    from anaties import helpers as helpy

I usually import plots as `aplots` and stats as `aplots`.   

Where `anaties_path` is the path to the anaties folder you downloaded (e.g., 'x/y/z/anaties/'). Later if you want to update the package, you can just do `git pull origin master` from within `anaties_path`.


## Useful sources
### Smoothing
- https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html

### FFT (this needs to be replaced with Welch)
- https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/
- https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html#sphx-glr-intro-scipy-auto-examples-plot-fftpack-py


## Notes
### Notes on FFT
Replace PSD with the spectrum package, or at least Welch's method:
https://pyspectrum.readthedocs.io/en/latest/

### Edge artifacts
Handling edge artifacts can be tricky: currently I use Gustaffson's method as the default, though at some point might tinker with that -- there are many options.

### What about wavelets?
I may add wavelets at some point, but it isn't plug-and-play enough for this repo. If you want to get started with wavelets in Python, I recommend http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/

### Tolerance values
For a discussion of the difference between relative and absolute tolerance values when testing floats for equality, for instance as used in `helpers.is_symmetric()`, see:
 https://stackoverflow.com/questions/65909842/what-is-rtol-for-in-numpys-allclose-function

## Acknowledgments
- Songbird wav is open source from: https://freesound.org/people/Sonic-ranger/sounds/243677/
- Developed with the support of NIH Bioinformatics, and the Neurobehavioral Core at NIEHS.

## To do
- mn_sem should work on 2d array if you give it an axis.
- Replace fft with welch it is *way* better for getting power spectrum.
- For spectrogram add denoising (e.g., 60hz) and filtering options.
- Do I want data-scroller or not?
- For freqhist should I guarantee it sums to 1 even when bin widths don't match data limits? Probably not. Something to think about though.
- In smoother, consider switching from filtfilt() to sosfiltfilt() for reasons laid out here: https://dsp.stackexchange.com/a/17255/51564
- Convert notch filter to sos?
- Make power spectrume stimation better than fft ffs (at *least* use welch):
https://github.com/cokelaer/spectrum
https://pyspectrum.readthedocs.io/en/latest/
- Add threshold to spectrogram plot?
- add ability to control event colors in spectrogram.
- ind_limits: add checks for data, data_limits, clarify description and docs
- Add audio playback of signals (see notes in audio_playback_workspace), incorporate this into some tests of filtering, etc.. simpleaudio package is too simple I think.
- Add numerical tests with random seed set not just graphical eyeball tests.
- Long-term: autodocs (sphinx?)
- Long-term: Make audio player that shows location in waveform for spectrogram.
