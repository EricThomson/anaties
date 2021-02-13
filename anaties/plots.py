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
    plot relative frequency histogram (instead of density or count). 
    
    Adapted from:
    https://github.com/matplotlib/matplotlib/issues/10398#issuecomment-366021979
   
    Inputs:
        data: 1d array of values
        bins: int or 1d array of bin edges (bin edges are half-open intervals [a, b) except the last which is [a,b] )
        color ('k'): color to paint bars
        
    Outputs:
        Draws frequency histogram such that all the binned data sum to 1 
          Will only sum to 1 if bin edges contain full range of values
        n: array of values of the histogram in the bins
        bin_edges: n+1 array of bin edges.
        
    To do: 
        - add checks on bins or outputs to guarantee they will sum to 1
        - Add example here
        
    """
    n, bin_edges, _ = plt.hist(data, bins, color = color, 
                       weights = np.ones(len(data))/len(data), density = False)
    return n, bin_edges


def rect_highlight(shade_range, orientation = 'vert', color = (1,1,0), alpha = 0.3):
    '''
    overlay transluscent highlight over current figure 
    
    Inputs:
        shade_range [min, max] range to draw highlights on
        orientation (str): 'vert' or 'horiz' for vertical/horizontal highlight
        color (rgb): color of bar (default (1,1,0) yellow)
        alpha (float): level of transparency (0.3)
        
    Outputs: none -- just adds rectangle to current figure.
    
    To do: 
        - add example here
    '''
    from matplotlib.patches import Rectangle
    shade_mag = shade_range[1]-shade_range[0]
    
    # vertically oriented bar
    if orientation == 'vert':
        y_ax = plt.gca().get_ylim()
        y_height = y_ax[1]-y_ax[0]
        rect_xy = (shade_range[0], y_ax[0])
        rect = Rectangle(rect_xy, width = shade_mag, height= y_height,
                         color = color, alpha = alpha)
    #horizontally oriented bar (for spectrogram etc)
    elif orientation == 'horiz': 
        x_ax = plt.gca().get_xlim()
        x_width = x_ax[1]-x_ax[0]
        rect_xy = (x_ax[0], shade_mag[0]) #x, y
        rect = Rectangle(rect_xy, width = x_width, height=shade_mag,
                         color = color, alpha = alpha)
    plt.gca().add_patch(rect)
    
    return





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
    
    """
    Test rect_highlight
    """
    print("anaties.plots: testing rect_highlight()...")
    plt.figure('Testing Highlighter')
    xdat = np.linspace(0.0, 100, 1000)
    ydat = np.sin(0.1*xdat)+np.random.normal(scale=0.1,size=xdat.shape)
    plt.plot(xdat, ydat, color = 'black', linewidth = 0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    rect_highlight([35,55], orientation = 'vert', color = (1,1,0), alpha = 0.8)
    plt.grid()

    
    
    
    
    
    
    
    
    
