import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.lse import load_lse_fest, AlphaEst
'''
Description:
Processing the instantaneous frquency estimates from
autoregressive.py and lse.py

Date: 
8/30/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


def check_fest(freq, proj_str='s5'):
    t, f = load_lse_fest(freq, proj_str=proj_str)
    delta_t = t[2]-t[0]
    ind= int(35*60 / delta_t)
    fig = plt.figure()
    plt.plot(t[:ind], f[:ind])
    plt.savefig(str(freq) + '_lse.png')
    plt.close(fig)
    return
    

check_fest(197, 'arcmfp1')
