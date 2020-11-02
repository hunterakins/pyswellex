import numpy as np
import sys
from matplotlib import pyplot as plt
from env.env.json_reader import write_json

'''
Description:

Date: 

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

def make_fest_conf(freq, exp_id):
    window_spacing = 3 # 1 minute spacing
    window_len = 47 # 43 minute fft length
    num_windows = 3
    track_chunks = [[3 + i*window_spacing, (3+window_len) + i*window_spacing] for i in range(num_windows)]
    svd = False
    mode = False
    window = False
    gammas= [0,1]

    conf_dict = {'freq': freq, 'track_chunks': track_chunks, 'svd': svd, 'mode' :mode, 'window': window, 'gammas': gammas, 'exp_id':exp_id}
    fname = 'confs/' + exp_id + '.conf'
    write_json(fname, conf_dict)

if __name__ == '__main__':
    freq = int(sys.argv[1])
    exp_id = sys.argv[2]
    make_fest_conf(freq, exp_id)
