"""
Tests for helpers in anaties package

https://github.com/EricThomson/anaties
"""
import numpy as np
from freezegun import freeze_time

from anaties import helpers


def test_datetime_string():
    """
    Use the freezegun package to set a particular time.
    Test in relation to that particular comparison string.
    For passing condition, fix datetime to your set value using freezegun.
    For failing condition, set it to now.
    """
    comparison_datestring = "_20120114_120001"
    dts1 = helpers.datetime_string()
    with freeze_time("2012-01-14 12:00:01"):
        dts2 = helpers.datetime_string()
    assert dts1 != comparison_datestring
    assert dts2 == comparison_datestring


def test_file_exists(tmp_path):
    """
    Use the temp_path fixture provided by pytest: it is a temporary
    Path directory that you can then use to generate a file.

    For passing condition, create a file. For failing condition, do not.
    """
    temp_filepath_pass = tmp_path / 'tmp.txt'
    temp_filepath_pass.write_text("content")
    temp_filepath_fail = tmp_path / 'foo.txt'
    assert helpers.file_exists(temp_filepath_pass)
    assert not helpers.file_exists(temp_filepath_fail)


def test_get_bins():
    """
    This is fairly straightforward, but at some point could add
    more thorough testing with different data types (ints)

    To add:
        precision none/not none
        remainder !=0:

    """
    # Precision None, remainder branch evaluates to 0
    bin_edges, bin_centers = helpers.get_bins(0, 20, bin_width=5)
    assert np.array_equal(bin_edges, np.array([0., 5., 10., 15., 20.]))
    assert np.array_equal(bin_centers, np.array([2.5, 7.5, 12.5, 17.5]))

    # Precision not None, remainder branch evaluates to 0

    # Precision not None, remainder evaluates to nonzero (throws exception)
    # something involving pytest.raises(ValueError)


def test_get_offdiag_vals():
    assert False


def test_ind_limits():
    foo_array = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    dat_lims = [2.5, 7]
    lim_inds = helpers.ind_limits(foo_array, data_limits=dat_lims)
    assert lim_inds == (2, 6)


def test_is_symmetric():
    assert False


def test_rand_rgb():
    assert False
