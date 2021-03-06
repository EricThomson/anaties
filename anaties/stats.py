"""
Statistical functions for anaties package

https://github.com/EricThomson/anaties
"""

import numpy as np



def med_semed(array, axis = None):
    """
    return median and std error of the median or numpy array
    """
    return np.median(array, axis = None), se_median(array, axis = None)


def mn_sem(array, axis = None):
    """
    return mean and std error of mean of a numpy array
    """
    return np.mean(array, axis = axis), se_mean(array, axis = axis)


def mn_std(array, axis = None):
    """
    return mean and std deviation of numpy array
    """
    return np.mean(array, axis = axis), np.std(array, ddof = 1)


def se_mean(array, axis = None):
    """
    calculate std error of mean of 1d numpy array
    """
    return np.std(array, axis = axis, ddof = 1)/np.sqrt(len(array))



def se_median(array, axis = None):
    """
    Calculate standard error of the median of 1d numpy array
    Uses the approximation 1.253*std_err_mean

    Adapted from:
    https://stats.stackexchange.com/a/196666/17624
    """
    return 1.2533*se_mean(array, axis = axis)


def cramers_v(test_stat, n, df):
    """
    cramer's v for calculating effect size of chi-square test.
    test_stat is chi-square statistic
    n is total number of observations
    df is df*  min((r-1)(c-1))

    output: V value

    Interpreting V will depend on degrees of freedom (for instance, more degrees of freeecom
    means smaller effect size is impressive).

    See:
    https://www.real-statistics.com/chi-square-and-f-distributions/effect-size-chi-square/
    """
    return np.sqrt(test_stat/(n*df))



if __name__ == '__main__':
    print("no tests written for anaties.stats yet. come on brah")
