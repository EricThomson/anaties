# anaties
An analysis utilities package. Common operations like signal smoothing that I find myself using in multiple projects.

Brief summary of utilities:  

    signals.py (for 1d data arrays, or arrays of such arrays)
        - smooth: smooth a signal with a window (gaussian, etc)
        - smooth_rows: smooth each row of a 2d array using smooth()
        - power_spec: get the power spectral density or power spectrum
        - spectrogram: calculate/plot spectrogram of a signal
        - notch_filter: notch filter to attenuate specific frequency (e.g. 60hz)
        - bandpass_filter: allow through frequencies within low- and high-cutoff

    plots.py (basic plotting)
        - error_shade: plot line with shaded error region
        - freqhist: calculate/plot a relative frequency histogram
        - paired_bar: bar plot for paired data
        - plot_with_events: plot with vertical lines to indicate events
        - rect_highlight: overlay rectangular highlight on current figure

    stats (basic statistical things)
        - med_semed: median and std error of median of an array
        - mean_sem: mean and std error of the mean of an array
        - mean_std: mean and standard deviation of an array
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
    from anaties import plots as aplots
    from anaties import stats as astats

Where `anaties_path` is the path to the anaties folder you downloaded (e.g., 'x/y/z/anaties/'). Later if you want to update the package, you can just do `git pull origin master` from within `anaties_path`.


## Useful sources
### Smoothing
- https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html

## Notes
### What about wavelets?
I may add wavelets at some point, but it isn't plug-and-play enough for this repo. If you want to get started with wavelets in Python, I recommend http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/

### Tolerance values
For a discussion of the difference between relative and absolute tolerance values when testing floats for equality (for instance as used in `helpers.is_symmetric()`) see:
 https://stackoverflow.com/questions/65909842/what-is-rtol-for-in-numpys-allclose-function

## Acknowledgments
- Songbird wav is open source from: https://freesound.org/people/Sonic-ranger/sounds/243677/
- Developed with the support from NIH Bioinformatics and the Neurobehavioral Core at NIEHS.

## To do
- Put everything into common namespace (__init__)
- finish plots.twinx make sure it works
- add test for plots.error_shade.
- Add return object for plots.rect_highlight()
- paired_bar (and mean_sem/std need to handle one point better: currently throwing warning)
- Add a proper suptitle fix in aplots it is a pita to add manually/remember:
      f.suptitle(..., fontsize=16)
      f.tight_layout()
      f.subplots_adjust(top=0.9)
- add proper documentation and tests to stats module.
- add ax return for plot functions, when possible.
- plots: add errorshade plot that acts as wrapper for fill_between.
- For freqhist should I guarantee it sums to 1 even when bin widths don't match data limits? Probably not. Something to think about though.
- In smoother, consider switching from filtfilt() to sosfiltfilt() for reasons laid out here: https://dsp.stackexchange.com/a/17255/51564
- Convert notch filter to sos?
- For spectral density estimation consider adding multitaper option. Good discussions:
https://github.com/cokelaer/spectrum
https://pyspectrum.readthedocs.io/en/latest/
https://mark-kramer.github.io/Case-Studies-Python/04.html
- add ability to control event colors in spectrogram.
- consider adding wavelets.
- ind_limits: add checks for data, data_limits, clarify description and docs
- Add audio playback of signals (see notes in audio_playback_workspace), incorporate this into some tests of filtering, etc.. simpleaudio package is too simple I think.
- Add numerical tests with random seed set not just graphical eyeball tests.
- Long-term: autodocs (sphinx?)
- Do I want data-scroller or not? Would prefer something better, maybe something like in pyqtgraph.
