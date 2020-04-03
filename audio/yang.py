import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy.signal import firwin, convolve, stft, detrend, decimate, get_window


"""
Implement the PLL detailed in Yang 2015 (Source depth estimation based on
synthetic aperture beamforming for a moving source)
"""


def bp_filter_dats(f0, sensor):
    f_filter = f0*(1 + 2.4/1500) # from Yang pf. 1681 after equation 7
    low_cut = f_filter - .5
    high_cut = f_filter + .5
    order,fs  = 1024, 1500
    filter_vals = firwin(order, [low_cut, high_cut], fs=fs, pass_zero=False, window='hann')
    vals = convolve(sensor[:,0], filter_vals, mode='same')
    return vals

if __name__ == '__main__':
    sensor = np.load('npy_files/sensor1.npy') # zero mean
    sensor -= np.mean(sensor)
    f0 = 49
    fdat = bp_filter_dats(f0, sensor)
    
    """ Consider a small section for testing purposes """
    N = 1500*1
    fdat = fdat[:N]
    plt.plot(fdat)
    plt.show()

