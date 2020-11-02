import numpy as np
from matplotlib import pyplot as plt
from swellex.ship.ship import get_good_ests, good_time
from scipy.signal import detrend, firwin, convolve, lombscargle, find_peaks
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
import sys
import json
from swellex.audio.tscc_demod_lib import fest_demod


'''
Description:
Use the ESPRIT instantaneous frequency estimates
to remove the DOppler broadening due to source accelreations. 
Scale the accelerations by gamma


Date: 

Author: Hunter Akins
'''

def make_proj_dir(exp_id, mode_filter, svd, window):
    proj_dir = exp_id
    if mode_filter == True:
        proj_dir += '_mode'
    if svd == True:
        proj_dir += '_svd'
    if window == True:
        proj_dir += '_window'
    proj_dir += '/'
    return proj_dir

def get_ref_freq(freq):
    """ Set the deep flag if frequency
    corresponds to a deep source band 
    Choose reference freq estimates accordingly"""
    deep_freqs = [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388]
    if freq in deep_freqs:
        deep = True
        ref_freq = 388
    else:
        deep = False
        ref_freq=385
    return deep, ref_freq
   
def parse_conf_dict(diction): 
    freq = diction['freq']
    exp_id = diction['exp_id']
    track_chunks = diction['track_chunks']
    gammas = diction['gammas']
    mode_filter_flag = diction['mode']
    svd_flag = diction['svd'] 
    window_flag = diction['window'] 
    deep_flag, ref_freq = get_ref_freq(freq)
    proj_dir = exp_id + '/'
    return freq, exp_id, track_chunks, gammas, mode_filter_flag, svd_flag, window_flag, deep_flag, ref_freq, proj_dir

if __name__ == '__main__':

    conf_file = sys.argv[1]
    with open('confs/' + conf_file, 'r') as f:
        lines = f.readlines()
        json_str = lines[0]
        diction = json.loads(json_str)

    freq, exp_id, track_chunks, gammas, mode_filter_flag, svd_flag, window_flag, deep_flag, ref_freq, proj_dir = parse_conf_dict(diction)

    folder_root = '/oasis/tscc/scratch/fakins/' + str(freq) + '/'
    if proj_dir[:-1] not in os.listdir(folder_root):
        os.mkdir(folder_root + proj_dir)
    folder_root += proj_dir
    
    print('Running fest_demod for freq ', freq, ' with track_chunks ', track_chunks)
    np.save(folder_root + 'gammas.npy', gammas)
    fest_demod(freq, gammas, lim_list=track_chunks, ref_freq=ref_freq, suffix='', folder_root=folder_root, deep=deep_flag, mode_filter=mode_filter_flag, svd=svd_flag, window=window_flag)
