import numpy as np
from matplotlib import pyplot as plt

'''
Description:
I should be use average kr, not k, so it may be slightly frequency dependent
Look at waveguide effect on the average group speed (for computing expected doppler
shift )

Date: 06/2020

Author: Hunter Akins
'''


freqs= [49, 64, 79, 94, 109, 112,127, 130, 148, 166, 201, 283, 338, 388, 145, 163, 198,232, 280, 335, 385, 10000, 20000] 

def bar_k(freq):
    """ Compute average kr for ideal waveguide of depth 215 meters """
    krs = []
    D = 215
    i = 0
    omeg = 2*np.pi*freq
    c = 1500
    k = omeg/c
    kz = i*i*np.pi*np.pi/D/D
    while kz < k:
        kr = np.sqrt(k*k - kz*kz)
        i += 1
        kz = i*i*np.pi*np.pi/D/D
        krs.append(kr)
    kbar = sum(krs)/len(krs)
    return kbar

ratios = []
for f in freqs:
    omeg = 2*np.pi* f
    k = omeg/1500
    kbar = bar_k(f)
    ratio = kbar/k
    ratios.append(ratio)

plt.scatter(freqs, ratios)
plt.show()
