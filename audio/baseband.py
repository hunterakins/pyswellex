import numpy as np
from matplotlib import pyplot as plt
from tscc_demod_lib import fest_baseband
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
    for freq in freqs:
        naive= False
        deep_flag, ref_freq = get_ref_freq(freq)
        print(deep_flag)
        start_min = 6.5 
        end_min = 45
        fest_baseband(freq, ref_freq=ref_freq, lim_list=[[6.5, 45]], deep=deep_flag,naive=naive) 
        naive= True
        deep_flag, ref_freq = get_ref_freq(freq)
        print(deep_flag)
        start_min = 5
        end_min = 40
        fest_baseband(freq, ref_freq=ref_freq, lim_list=[[6.5, 45]], deep=deep_flag,naive=naive) 

