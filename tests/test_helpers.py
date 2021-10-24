"""
Tests for helpers in anaties package

https://github.com/EricThomson/anaties
"""

import numpy as np

from anaties import helpers 

  
def test_ind_limits():
    foo_array = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    dat_lims = [2.5, 7]
    lim_inds = helpers.ind_limits(foo_array, data_limits = dat_lims)
    assert lim_inds == (2, 6)

    