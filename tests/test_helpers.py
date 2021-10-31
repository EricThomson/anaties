"""
Tests for helpers in anaties package

https://github.com/EricThomson/anaties
"""
import os
import numpy as np
import tempfile

from anaties import helpers


def test_datetime_string():
    assert False


def test_file_exists(tmp_path):
    """
    NOPE: do this: https://docs.pytest.org/en/6.2.x/tmpdir.html
    This test uses a couple of tricks.

    First, create a context manager that generates a temporary directory
    that will garbage collect the directory and all its contents after exiting
    the block.

    Second, hack a touch command with the open(temp_filepath) trick.

    https://stackoverflow.com/a/10940847/1886357
    https://stackoverflow.com/a/12654798/1886357
    """
    temp_filepath_pass = tmp_path / 'tmp.txt'
    temp_filepath_pass.write_text("content")
    assert helpers.file_exists(temp_filepath_pass)


def test_get_bins():
    assert False


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
