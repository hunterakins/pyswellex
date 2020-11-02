import numpy as np
from matplotlib import pyplot as plt
from swellex.ship.ship import get_good_ests, good_time
from scipy.signal import detrend, firwin, convolve, lombscargle, find_peaks
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
import sys
import json
from tscc_demod_lib import mode_filter


'''
Description:
Apply a mode filter to the data for the frequency band supplied
in the conf file

Date:  7/15

Author: Hunter Akins
'''


if __name__ == '__main__':
    deep_freqs = [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388]
    #test_script()


    #x = np.load('npy_files/49_mini.npy')
    #save_demod(49, 1, deep=True)

    config_file = sys.argv[1]

    with open('confs/' + config_file, 'r') as f:
        lines = f.readlines()
        json_str = lines[0]
        diction = json.loads(json_str)
    freq = diction['freq']
    gamma = diction['gamma']
    chunk_len = diction['chunk_len']
    print(freq, gamma, chunk_len)
    deep = False
    if freq in deep_freqs:
        deep = True
    track_chunks = [[0, 15], [10, 25], [20,35],[30, 45], [40,55]]
    for i, track_chunk in enumerate(track_chunks):
        mode_filter(freq, deep=deep, chunk_len=chunk_len,start_min=track_chunk[0], end_min=track_chunk[1], prefix='filter_chunk_'+str(i) +'_')
