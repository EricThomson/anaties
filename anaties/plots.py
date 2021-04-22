"""
Plotting functions for anaties package

https://github.com/EricThomson/anaties
"""
import sys
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(str(Path('.').absolute().parent))
import anaties.helpers as helpy
import anaties.stats as astats


def error_shade(x, y, error_mag, line_color='k', line_width=1, shade_color='gray', alpha=0.3, ax = None):
    """
    plot x and y line +/- shaded error region.
    Currently provides minimal properties for changing line (color and linewidth)
    
    Inputs:
        x: 1d array of x values
        y: 1d array of y values
        error_mag: 1d array of error values
        line_color ('k'): color for main line
        line_width (1): width of line
        shade_color ('gray'): color for shaded error region
        alpha (0.3): transparency of shaded region
        ax (None): axes object if you are plotting it in another figure
        
    outputs:
        axes object for plot
    
    """
    if ax is None:
        f, ax = plt.subplots()
    ax.fill_between(x, y+error_mag, y-error_mag, alpha = alpha, color = shade_color)
    ax.plot(x, y, color = line_color, linewidth = line_width)
    return ax
    

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


def paired_bar(xdata, ydata, xrange, xbin_width, axlabels = ['x', 'y'], plot_on = 1):
    """
    Create bar plot for paired x/y data, especially useful when x is a float.
    
    Given paired data (xdata, ydata), and desired range and bin-width
    for xdata, show mean/sem of corresponding ydata. Optionally plots
    bar plot of y mn/sem bins vs x and scatter plot of original data.
       
    Inputs:
        xdata: 1d numpy array
        ydata: 1d numpy array
        xrange [min, max] min and max bounds for x data for binning
        xbin_width (float): bin width you want for binning x data
        plot_on (int): 0: no plots, 1 bar plot, 2 bar+scatter
    
    Outputs:
        Returns simple object, bar_properties, a class with following attributes:
            xbin_edges: array of bin edges formed from input params
            xbin_centers: num_bins length array of bin centers between xbin_edges
            ybinned_means: num_bins length array of mean value of ydata in points corresponding to binned xdata
            ybinned_sems: num_bins length array of sem of ydata in points corresponding to binned xdata
            ybinned_data: num_bins length list of array of y data in each bin corresponding to xbin_centers
            num_samples: num_bins length list of number of points in each bin
            axes: axes of plots if you are plotting, None if plot_on=0
    
    Example (weight vs height):
        height_dat = np.asarray([60, 60.1, 59.2, 60.3, 60.4,
                                 65, 65.1, 66.2, 65.3, 65.4,
                                 70, 70.1, 70.2, 70.3, 70.4])  
        weight_dat = np.asarray([120, 100, 140, 150, 90,
                                 140, 120, 161, 170, 110,
                                 160, 140, 180, 190, 130])
        weight_bar = paired_bar(height_dat, weight_dat, 
                                xrange = [57.5, 72.5], xbin_width = 5, plot_on = 2,
                                axlabels = ['Height (inches)', 'Weight (lbs)'])  
        weight_bar.axes[1].set_ylim(100, 180);
        weight_bar.axes[0].set_ylim(75, 205)
        weight_bar.axes[0].set_xlim(58.5,71.5)
        weight_bar.axes[1].set_xticks([60, 65, 70]);
                         
    Note:
        For categorical x data, plt.bar() is often fine. 
        This is for more complex cases where you don't feel 
        like setting up bin edges beforehand.    
    """
    if xdata.size != ydata.size:
        raise ValueError('xdata and ydata are paired data: they must be the same size')
        
    xbin_edges, xbin_centers = helpy.get_bins(xrange[0], xrange[1], bin_width = xbin_width)
    num_bins = len(xbin_centers)
    # for each element of xdata, get its bin number between 1...num_bins
    xdata_inds = np.digitize(xdata, xbin_edges, right = True) 
    
    # use those some indices to extract values from ydata and insert into bins
    ybinned_means = []
    ybinned_sems = []
    ybinned_data = []
    num_samples = []
    for bin_num in range(num_bins):
        current_ydata = ydata[xdata_inds == bin_num+1]
        num_samples_bin = len(current_ydata)
        num_samples.append(num_samples_bin)
        ybinned_data.append(current_ydata)
        if num_samples_bin > 1:
            y_mn, y_sem = astats.mean_sem(current_ydata)
        else:
            y_mn = np.nan
            y_sem = np.nan
        ybinned_means.append(y_mn)
        ybinned_sems.append(y_sem)
    ybinned_means = np.asarray(ybinned_means)
    ybinned_sems = np.asarray(ybinned_sems)
        
    if plot_on == 1:
        f, ax = plt.subplots()
        ax.bar(xbin_centers, ybinned_means, 
                width = 0.94*xbin_width, 
                yerr = ybinned_sems, 
                color = 'k',
                zorder = 3);
        ax.set_xlabel(axlabels[0])
        ax.set_ylabel(axlabels[1])
    elif plot_on == 2:
        f, (ax1, ax2) = plt.subplots(2,1)
        ax1.scatter(xdata, ydata, color = 'k')
        ax1.set_xlabel(axlabels[0])
        ax1.set_ylabel(axlabels[1])
        
        ax2.bar(xbin_centers, ybinned_means, 
                width = 0.94*xbin_width, 
                yerr = ybinned_sems, 
                color = 'k',
                zorder = 3);
        ax2.set_xlabel(axlabels[0])
        ax2.set_ylabel(axlabels[1])
        plt.tight_layout()

    #create simple namespace to contain all the bits to return:
    bar_properties = SimpleNamespace()
    bar_properties.xbin_edges = xbin_edges
    bar_properties.xbin_centers = xbin_centers
    bar_properties.ybinned_means = ybinned_means
    bar_properties.ybinned_sems = ybinned_sems
    bar_properties.ybinned_data = ybinned_data
    bar_properties.num_samples = num_samples #number of samples in each bin
    if plot_on == 0:
        bar_properties.axes = None
    elif plot_on == 1:
        bar_properties.axes = (ax,)
    elif plot_on == 2:
        bar_properties.axes = (ax1, ax2)
        
    return bar_properties 


def plot_with_events(x, y, linewidth = 0.5, color = 'black', event_linewidth = 1, 
                     all_events = None, event_colors = None, ax = None):
    """
    Plot a signal and events...in technicolor!
    
    Inputs:
        x: array for x axis of plot
        y: array for y axis of plot
        linewidth for plot (0.5)
        color for plot line (black)
        all_events: list of arrays (None)
        event_colors: list of colors (None)
        ax (axes): axes object to paint upon (None)
    Output:
        axes object so you can have fun
    """
    if ax is None:
        f, ax = plt.subplots()
    if event_colors is not None:
        for event_ind, events in enumerate(all_events):
            for event in events:
                ax.axvline(event, color=event_colors[event_ind], 
                           linewidth = event_linewidth, zorder=3)
    ax.plot(x, y, linewidth = linewidth, color = color)
    ax.autoscale(enable=True, axis='x', tight=True)
    return ax


def rect_highlight(shade_range, orientation = 'vert', color = (1,1,0), alpha = 0.3, ax = None):
    '''
    overlay transluscent rectangular highlight over figure
    
    Inputs:
        shade_range [min, max] range to draw highlights on
        orientation (str): 'vert' or 'horiz' for vertical/horizontal highlight
        color (rgb): color of bar (default (1,1,0) yellow)
        alpha (float): level of transparency (0.3)
        ax (axes): axes object to paint upon (None)
        
    Outputs: none -- just adds rectangle to current figure or axes input.
    
    To do:
        have it return axes so people can work with them!
    
    Notes
        - If you ever want to add multiple rects to different axes:
          https://stackoverflow.com/questions/47554753/can-not-put-single-artist-in-more-than-one-figure
    '''
    from matplotlib.patches import Rectangle
    shade_mag = shade_range[1]-shade_range[0]
    
    # vertically oriented bar
    if orientation == 'vert':
        if ax is not None:
            y_ax = ax.get_ylim()
        else:
            y_ax = plt.gca().get_ylim()
        y_height = y_ax[1]-y_ax[0]
        rect_xy = (shade_range[0], y_ax[0])
        rect = Rectangle(rect_xy, width = shade_mag, height= y_height,
                         color = color, alpha = alpha)
    #horizontally oriented bar (for spectrogram etc)
    elif orientation == 'horiz': 
        if ax is not None:
            x_ax = ax.get_xlim()
        else:
            x_ax = plt.gca().get_xlim()
        x_width = x_ax[1]-x_ax[0]
        rect_xy = (x_ax[0], shade_range[0]) #x, y
        rect = Rectangle(rect_xy, width = x_width, height=shade_mag,
                         color = color, alpha = alpha)
        
    if ax is not None:
        ax.add_patch(rect)
    else:
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
    plt.xlabel('Incindiery Intensity')
    plt.ylabel('relative frequency')
    plt.gcf().canvas.set_window_title("Testing freqhist()")
    
    
    """
    Test paired_bar
    """
    print("anaties.plots: testing paired_bar()...")
    height_dat = np.asarray([60, 60.1, 59.2, 60.3, 60.4,
                             65, 65.1, 66.2, 65.3, 65.4,
                             70, 70.1, 70.2, 70.3, 70.4])  
    weight_dat = np.asarray([120, 100, 140, 150, 90,
                             140, 120, 161, 170, 110,
                             160, 140, 180, 190, 130])
    weight_bar = paired_bar(height_dat, weight_dat, 
                            xrange = [57.5, 72.5], xbin_width = 5, plot_on = 1,
                            axlabels = ['Height (inches)', 'Weight (lbs)'])  
    plt.ylim(75, 205)
    plt.xlim(57.5,72.5)
    plt.xticks([60, 65, 70])
    plt.gcf().canvas.set_window_title("Testing paired_bar()")

    """
    Test plot_with_events
    """
    print("anaties.plots: testing plot_with_events()...")
    x = np.linspace(0, 20, 100)
    y = np.sin(x) + np.random.normal(scale = 0.2, size=x.shape)
    sounds = [5, 10, 15]
    lights = [7, 12, 17]
    ax = plot_with_events(x, y, all_events = [sounds, lights], event_colors = ['red', 'green'])
    plt.gcf().canvas.set_window_title("Testing plot_with_events()")
    
    """
    Test rect_highlight
    """
    print("anaties.plots: testing rect_highlight()...")
    plt.figure('Testing rect_highlight()')
    xdat = np.linspace(0.0, 100, 1000)
    ydat = np.sin(0.1*xdat)+np.random.normal(scale=0.1,size=xdat.shape)
    plt.plot(xdat, ydat, color = 'black', linewidth = 0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    rect_highlight([35,55], orientation = 'vert', color = (1,1,0), alpha = 0.8)
    plt.grid()

    plt.show()
    
    
    
    
    
    
    
    
    
