import numpy as np
from matplotlib import pyplot as plt
from tscc_demod_lib import fest_baseband, vest_baseband
from lse import AlphaEst
from fest_demod import get_ref_freq
import sys

'''
Description:
Complex baseband the data using the 
reference frequency doppler broadening

Date: 

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


if __name__ == '__main__':
    freqs = [49, 64, 79, 94, 109, 127, 130, 145, 148, 163, 166]#, 198, 201, 232]
    freqs = [49]
    gamma = np.arange(0.9, 1.1, .02)
    gamma = np.insert(gamma, 0, 0)
    for freq in freqs:
        deep_flag, ref_freq = get_ref_freq(freq)
        print(deep_flag)
        start_min = 5
        end_min = 54
        vest_baseband(freq, ref_freq=ref_freq, lim_list=[[start_min, end_min]], deep=deep_flag,gamma=gamma) 
