"""
generic helper functions for anaties package

https://github.com/EricThomson/anaties
"""
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings


def datetime_string():
    """
    create a datetime string of form:  _YYYYMMDD_HHMMSS year month day _ hour minute second)
    useful for creating filenames

    Inputs: None

    Outputs:
        Single string of form' '_YYYYMMDD_HHMMSS' (year month day _ hour minute second)

    To do:
        add option to supress leading underscore.
    """
    return datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")


def file_exists(filepath):
    """
    Check to see if the file specified in filepath exists
    A pure wrapper for os.path.isfile()

    Inputs:
        filepath: string

    Output:
        bool (True if file exists, False if it does not)
    """
    return os.path.isfile(filepath)


def get_bins(min_edge, max_edge, bin_width=1, suppress_warning=False):
    """
    return array of bin edges/bin centers, given min/max values and bin_width

    Note:
    Creates floor(max_edge-min_edge)/bin_width bins.
    Throws warning if bin width does not evenly divide range of values.

    Inputs:
        min_edge: scalar lower bound
        max_edge: scalar upper bound
        bin_width: scalar bin width to divide up the range
        suppress_warning: boolean suppress warning for not evenly divisible range

    Outputs:
        bin_edges: 1d array
        bin_centers: 1d array

    Example:
        bin_edges, bin_centers = get_bins(0, 20, bin_width = 5)
        #to generate warning: 0, 4.5, 1/3

    Notes:
        bin_centers vectorization from: https://stackoverflow.com/a/23856065/1886357
    """
    if min_edge >= max_edge:
        raise ValueError("get_bins(): max_edge must be larger than min_edge")
    val_range = max_edge - min_edge
    num_bins = val_range/bin_width

    # make sure the bin_width evenly divides up the range of values
    if not suppress_warning:
        bin_remainder = num_bins % 1
        if bin_remainder != 0:
            msg1 = "get_bins(): range not evenly divisible by bin_width."
            msg2 = f"\nrange = {val_range}, bin width = {bin_width}"
            warning_message = msg1 + msg2
            warnings.warn(warning_message)

    num_bins = int(num_bins)
    num_edges = num_bins+1
    bin_edges = np.linspace(min_edge, max_edge, num_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_edges, bin_centers


def get_offdiag_vals(array):
    """
    For a symmetric array of values, pull the lower diagonal values
    (not including the diagonal elements), and indices. Useful for
    things like arrays of pairwise correlation/distances when you
    want to extract the unique values but not diagonal.

    Inputs:
        array: nxn symmetric matrix of values.

    Outputs:
        values: M = (n^2-n)/2-element array of values in the original 2d array
        indices: (row_inds, col_inds) tuple containing M-ary array of indices

    Notes
        To reconstruct values from array: array[indices]

    Adapted from:
        https://stackoverflow.com/a/44395030/1886357
    """
    if not is_symmetric(array):
        raise ValueError("get_offdiag_vals() requires symmetric array")
    num_rows = array.shape[0]
    # get indices of lower diag, not inclucing diagonal
    offdiag_indices = np.tril_indices(num_rows, k=-1)
    offdiag_values = array[offdiag_indices]
    return offdiag_values, offdiag_indices


def ind_limits(data, data_limits=None):
    """
    Given increasing data, and two data limits (min and max), returns indices
    such that data is between those limits (inclusive).

    inputs:
        data: nondecreasing 1d np array
        data_limits (2elt list): limits of data you want to select (if None just return 0, -1)
    outputs:
        first_ind: index of smallest data >= data_limits[0]
        last_ind: index of largest data <= data_limits[1]

    Notes
        When using the inds in slices, be sure you aren't missing the last one.

    To do:
        Add checks for 1d data, increasing data, data_limits.
    """
    first_ind = 0
    last_ind = -1
    if data_limits is not None:
        inds = np.where((data >= data_limits[0]) & (data <= data_limits[1]))[0]
        first_ind = inds[0]
        last_ind = inds[-1]
    return first_ind, last_ind


def is_symmetric(array, abs_tol=1e-8, rel_tol=1e-5):
    """
    Test if a numpy array is symmetric, within tolerance levels (is A = A')
    If either tolerance val is set to None, then no tolerance is used and
    uses strict equality (good for integer or other non-float array).

    Inputs:
        array: nxm numpy array (only nxn can be symmetric)

    Returns:
        symmetric (boolean): is matrix symmetric or not?

    Adapted from:
        https://stackoverflow.com/a/42913743/1886357
        https://stackoverflow.com/a/65909907/1886357
    """
    if abs_tol is None or rel_tol is None:
        symmetry = np.array_equal(array, array.T, equal_nan=True)
    else:
        symmetry = np.allclose(array, array.T, rtol=abs_tol, atol=abs_tol, equal_nan=True)

    return symmetry


def rand_rgb(num_vals):
    """
    return random rgb vals between 0-1 in Nx3 numpy array.
    Can be fed to plt.scatter c keyword to generate random colors.

    inputs:
        num_vals (int): number of rgb values wanted

    Outputs:
        rgb_vals: num_vals x 3 numpy array. To access element i: rgb_vals[i,]
    """
    rgb_vals = np.random.rand(num_vals, 3)
    return rgb_vals


if __name__ == '__main__':
    print("\nTesting anaties.helpers...")
    """
    Test datetime_string()
    """
    print("\nanaties.helpers: testing datetime_string()...")
    print(f"string generated: {datetime_string()}")

    """
    Test get_bins()
    """
    print("\nanaties.helpers: testing get_bins()...")
    bin_edges, bin_centers = get_bins(0, 3, bin_width=0.5)
    print("min, max, width: 0, 3, 0.5")
    print(f"Bin edges: {bin_edges}\nBin centers: {bin_centers}")

    """
    Test get_offdiag_vals()
    """
    print("\nanaties.helpers: testing get_offdiag_vals()...")
    test_mat = np.asarray([[1, 2, 3, 4],
                           [2, 5, 7, 8],
                           [3, 7, 5, 12],
                           [4, 8, 12, 5]])
    vals, inds = get_offdiag_vals(test_mat)
    print(f"Array:\n{test_mat}")
    print(f"Values: {vals}")
    print(f"Inds:\nRow:{inds[0]}\nCol:{inds[1]}")

    """
    Test ind_limits
    """
    print("\nanaties.helpers: testing ind_limits()...")
    foo_array = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    dat_lims = [2.5, 7]
    lim_inds = ind_limits(foo_array, data_limits=dat_lims)
    assert lim_inds == (2, 6), print("ind_limits fail")
    print(f"inds containing {dat_lims} within {foo_array}:\n{lim_inds}")

    """
    Test is_symmetric
    """
    print("\nanaties.helpers: testing is_symmetric()...")
    test_mat = np.asarray([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]])
    symm_output = is_symmetric(test_mat)
    print(f"Array:\n{test_mat}\nSymmetry: {symm_output}")

    """
    test rand_rgb
    """
    print("\nanaties.helpers: testing rand_rgb()...")
    print("Generating a plot...")
    num_points = 20
    x_vals = np.arange(num_points)
    y_vals = x_vals**2
    plt.scatter(x_vals, y_vals, c=rand_rgb(num_points))
    plt.title('rand_rgb() test -- do you see  random colors?')
    plt.show()

    print("\nanaties.helpers: tests done...")


#
#
#
