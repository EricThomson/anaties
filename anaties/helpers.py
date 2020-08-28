"""
generic helper functions for anaties package

https://github.com/EricThomson/anaties
"""
import sys
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path('.').absolute().parent))


def datetime_string():
    """
    create a datetime string of form _YYYYMMDD_HHMMSS (year month day _ hour minute second)
    useful for creating filenames
    """
    return datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")


def ind_limits(data, data_limits = None):
    """ 
    Given increasing data, and two data limits (min and max), returns indices 
    such that data is between those limits (inclusive).
    
    inputs:
        data: nondecreasing 1d np array
        data_limits (2elt list): limits of data you want to select (if None just return 0, -1)
    outputs:
        first_ind: index of smallest data >= data_limits[0]
        last_ind: index of largest data <= data_limits[1]
    
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


def rand_rgb(num_vals):
    """
    return random rgb vals between 0-1 in Nx3 numpy array. 
    inputs:
        num_vals (int): number of rgb values wanted
    outputs:
        rgb_vals: num_vals x 3 numpy array. To access element i: rgb_vals[i,]
    """
    rgb_vals = np.random.rand(num_vals,3)
    return rgb_vals


if __name__ == '__main__':
    print("\nTesting anaties.helpers...")
    """
    Test datetime_string
    """
    print("\nanaties.helpers: testing datetime_string()...")
    print(f"string generated: {datetime_string()}")
    
    """
    Test ind_limits
    """
    print("\nanaties.helpers: testing ind_limits()...")
    foo_array = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    dat_lims = [2.5, 7]
    lim_inds = ind_limits(foo_array, data_limits = dat_lims)
    assert lim_inds == (2, 6), print("ind_limits fail")
    print(f"\tinds containing {dat_lims} within {foo_array}: {lim_inds}")
    

    """
    test get_rgb
    """
    print("\nanaties.helpers: testing rand_rgb()...")
    num_points = 20
    x_vals = np.arange(num_points)
    y_vals = x_vals**2
    plt.scatter(x_vals, y_vals, c = rand_rgb(num_points))
    plt.title('rand_rgb() test -- do you see a rainbow?')
 

    print("\nanaties.helpers: tests done...")   


#
    