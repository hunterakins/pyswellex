import numpy as np
from matplotlib import pyplot as plt
from swellex.ship.ship import get_good_ests, good_time
from scipy.signal import detrend, firwin, convolve, lombscargle, find_peaks
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
import sys
import json
from tscc_demod_lib import fest_demod


'''
Description:
Use the ESPRIT instantaneous frequency estimates
to remove the DOppler broadening due to source accelreations. 
Scale the accelerations by gamma


Date: 

Author: Hunter Akins
'''


if __name__ == '__main__':

    freq = int(sys.argv[1])
    gammas = np.arange(0.8, 1.2, .001)
    track_chunks = [[40+5*i, 40 + 5*i+5] for i in range(2)]
    folder_root = '/oasis/tscc/scratch/fakins/' + str(freq) + '/five_min_B/'
    fest_demod(freq, gammas, lim_list=track_chunks, ref_freq=385, suffix='', folder_root=folder_root)
