"""
Statistical functions for anaties package

https://github.com/EricThomson/anaties
"""

import numpy as np


def collective_correlation(x):
    """
    get collective correlation coefficient (aka generalized correlation)

    input:
        x: data array nxm (n observations, m variables)

    outputs:
        corr_coeff: correlation coefficient

    Notes:
       - Calculated as discussed in stack exchange:
          https://math.stackexchange.com/a/3795338/52369
       - If you get over/underflow errors for det(x), switch to np.slogdet()
    """
    # first get covariance matrix of x (ddof returns unbiased estimate)
    cov_mat = np.cov(x, ddof=1, rowvar=False)
    # product of all variances
    variance_product = cov_mat.diagonal.prod()

    cov_det = np.linalg.det(cov_mat)

    return np.sqrt(1 - cov_det/variance_product)


def med_semed(array, axis=None):
    """
    return median and std error of the median or numpy array
    """
    return np.nanmedian(array, axis=axis), se_median(array, axis=axis)


def mean_sem(array, axis=None):
    """
    return mean and std error of mean of a numpy array
    """
    return np.nanmean(array, axis=axis), se_mean(array, axis=axis)


def mean_std(array, axis=None):
    """
    return mean and std deviation of numpy array
    """
    return np.nanmean(array, axis=axis), np.nanstd(array, axis=None, ddof=1)


def se_mean(array, axis=None):
    """
    calculate std error of mean of a numpy array
    Sample standard deviation divided by the square root of number of samples

    only works for 1d or 2d arrays

    todo: improve documentation here of numels calc, and ddof
    """
    if array.ndim > 2:
        raise ValueError("se_mean only accepts 1d or 2d arrays")
    if axis is None:
        numels = len(array)
    elif axis == 0:
        numels = array.shape[1]
    elif axis == 1:
        numels = array.shape[0]
    return np.nanstd(array, axis=axis, ddof=1)/np.sqrt(numels)


def se_median(array, axis=None):
    """
    Calculate standard error of the median of a numpy array
    Uses the approximation se_median = 1.253*se_mean

    Adapted from:
    https://stats.stackexchange.com/a/196666/17624
    """
    return 1.2533*se_mean(array, axis=axis)


def cramers_v(test_stat, n, df):
    """
    cramer's v for calculating effect size of chi-square test.
    test_stat is chi-square statistic
    n is total number of observations
    df is df* =  min((r-1)(c-1))

    output: V value

    Interpreting V depends on degrees of freedom (for instance, higher
    dof means smaller effect size can still be "interesting" effect).

    See:
    https://www.real-statistics.com/chi-square-and-f-distributions/effect-size-chi-square/
    """
    return np.sqrt(test_stat/(n*df))


if __name__ == '__main__':
    print("no tests written for anaties.stats yet. come on bruv")
