"""
Plotting functions for anaties package

https://github.com/EricThomson/anaties
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path('.').absolute().parent))


def freqhist(data, bins, color = 'k'):
    """
    plot relative frequency histogram (instead of density or count). Wrapper for
    matplotlib hist with tweaks for freq hist. Based on discussion at github:
    https://github.com/matplotlib/matplotlib/issues/10398#issuecomment-366021979
   
    Inputs:
        data: 1d array of values
        bins: int or 1d array of bin edges (bin edges are half-open intervals [a, b) except the last which is [a,b] )
        color ('k'): color to paint bars
        
    Outputs:
        Draws frequency histogram such that all the binned data would sum to 1 
          Will only sum to 1 if bin edges contain full range of values
        n: array of values of the histogram in the bin edges
        bin_edges: n+1 array of bin edges.
        
    """
    n, bin_edges, _ = plt.hist(data, bins, color = color, 
                       weights = np.ones(len(data))/len(data), density = False)
    return n, bin_edges





#%%  run some tests
if __name__ == '__main__':
    plt.close('all')
    """
    Test freqhist
    """
    print("anaties.plots: testing freqhist()...")
    data = np.asarray([0.25, 0.25, 0.75, 1.25, 1.25, 1.25, 1.75, 1.75, 1.75, 1.75])
    bins = [0, 0.5, 1, 1.5, 2]
    n = freqhist(data, bins, color = 'g')
    plt.grid(axis = 'y')
