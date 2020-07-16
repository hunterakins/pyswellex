import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.wkb_approx import load_mode_krs

'''
Description:
Some simple routines to look at aperture size needed to 
resolve the  wvaenumbers

Date: 
7/14/2020

Author: Hunter Akins
'''


def plot_aperture_size(freq):
    krs = load_mode_krs(freq)
    diffs = krs[1:] - krs[:-1]
    ranges = 2*np.pi/(abs(diffs))
    plt.plot(krs[:-1], ranges)
    plt.show()

plot_aperture_size(49)
