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
    num_modes =int(sys.argv[2])
    deep_freqs = [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388]
    if freq in deep_freqs:
        deep = True
        ref_freq = 388
    else:
        deep = False
        ref_freq=385
    gammas = np.arange(0.8, 1.2, .001)
    gammas = [0,1]
    #track_chunks = [[40+5*i, 40 + 5*i+5] for i in range(2)]
    #track_chunks = [[0, 50]]
    track_chunks = [[45, 53]]
    proj_dir = '180m/'
    folder_root = '/oasis/tscc/scratch/fakins/' + str(freq) + '/'
    if proj_dir[:-1] not in os.listdir(folder_root):
        os.mkdir(folder_root + proj_dir)
    folder_root += proj_dir
    
    np.save(folder_root + 'gammas.npy', gammas)
    fest_demod(freq, gammas, lim_list=track_chunks, ref_freq=ref_freq, suffix='', folder_root=folder_root, deep=deep, num_modes=num_modes)
