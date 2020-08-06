import numpy as np
from matplotlib import pyplot as plt

'''
Description:
Using the rough formula derived in the notes
B(mode) = f_s ( 1/cb - 1/cw) v_s

Date: 8/1

Author: Hunter Akins
'''


def B(freq):
    return freq*(1/1450 - 1/1700)*3 


def B_sim(freq):
    krs = np.load('npy_files/' + str(freq) + '_krs.npy')
    return (np.max(krs) - np.min(krs)) * 3 / (2 *np.pi)


freqs = [109, 127, 145, 163, 198, 232, 280, 335, 385]

freqs = freqs + [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388]
for freq in freqs:
    plt.scatter(freq, B(freq), color='g')
    plt.scatter(freq, B_sim(freq), color='r')
plt.legend(['Approximate upper bound', 'Simulation using known SwellEx environment'])
plt.ylabel('Modal spread (Hz)')
plt.xlabel('Source frequency (Hz)')
plt.show()
    

