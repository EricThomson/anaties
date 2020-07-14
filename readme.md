# anaties
An analysis utilities package.

This is where I will put stuff I use all the time for processing signals, images, etc.. It will grow over time right now it includes basic signal processing stuff. 


## Install
Eventually I will build a way to install but for now:

    conda create -n anaties
    conda activate anaties
    conda install python=3.7
    conda install scipy numpy matplotlib
    conda install -c conda-forge opencv=4

Optional stuff -- I install spyder. Eventually I might make notebooks in which case I'd install jupyter and nodejs.

## Structure
Utilities are broken up into groups
anaties/
    signals.py  -- doing stuff with 1d signals
    #images.py  -- doing stuff with 2d images -- not added yet


## Notes on design decisions
###  Why no gaussian filter?
In `signals.py `I don't have a gaussian filter by default because I wanted to include relatively simple smoothing windows without needing to specify parameters, just the filter width, not additional filter parameters like sigma. At some point I may add a gaussian (just use `np.random.normal()`) but it is complicated enough that I will need to take a half day to do it right

### Edge artifacts
Handling edge artifacts can be tricky: you can pad it (with different parameters), and use Gustafsson's method. I like Gustaffson's method so went with that as the default. At some point I might tinker with that: again that will be a half day to really get it right. Frankly the decisions you make about your edges shouldn't make much difference: if they do something has probably gone wrong with your design at a previous step.

  :)
