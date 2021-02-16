"""
Statistical functions for anaties package

https://github.com/EricThomson/anaties
"""

import numpy as np




def mn_sem(array):
    """
    return mean and std error of mean of 1d numpy array
    """
    return np.mean(array), se_mean(array)


def mn_std(array):
    """
    return mean and std deviation of 1d array of data
    """
    return np.mean(array), np.std(array)


def se_mean(array):
    """
    calculate std error of mean of 1d numpy array 
    """
    return np.std(array)/np.sqrt(len(array))


def se_median(array):
    """
    Calculate standard error of the median of 1d numpy array
    Uses the approximation 1.253*std_err_mean
    
    Adapted from:
    https://stats.stackexchange.com/a/196666/17624    
    """
    return 1.2533*se_mean(array)


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