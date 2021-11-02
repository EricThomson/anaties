"""
Packaging for anaties

https://github.com/EricThomson/anaties
"""

from .signals import smooth
from .signals import smooth_rows
from .signals import power_spec
from .signals import spectrogram
from .signals import notch_filter
from .signals import bandpass_filter

from .plots import error_shade
from .plots import freqhist
from .plots import paired_bar
from .plots import plot_with_events
from .plots import rect_highlight
from .plots import vlines

from .stats import med_semed
from .stats import mean_sem
from .stats import mean_std
from .stats import se_mean
from .stats import se_median
from .stats import cramers_v

from .helpers import datetime_string
from .helpers import file_exists
from .helpers import get_bins
from .helpers import get_offdiag_vals
from .helpers import ind_limits
from .helpers import is_symmetric
from .helpers import rand_rgb
