import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.tscc_demod_lib import get_mode_filter_name, get_bb_name
from swellex.audio.config import get_proj_tones

'''
Description:
Apply the mode filters estimated in extract_modes to compress the basebanded data to 
num_modes channels

Date: 
9/18/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

freqs = get_proj_tones('arcmfp1')
gammas = np.arange(0.95, 1.05, 0.001)
for freq in freqs:
    mode_filter = np.load(get_mode_filter_name(freq, proj_str))
    for g in gammas:
        bb_name = get_bb_name(freq, proj_string, gamma)
        with open(bb_name, 'rb') as f:
            bb = pickle.load(f)
            bb.x = mode_filter@bb.x
            bb.save(proj_string, mf=True)
            



