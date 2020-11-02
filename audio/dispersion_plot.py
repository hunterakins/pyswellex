import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.signal import find_peaks
'''
Description:
Plot the dispersion curve for the estimated k_r

Date: 
7/23/2020

Author: Hunter Akins
'''



def load_psd():
    folders= os.listdir('npy_files/disp_plot')
    freqs = [int(x) for x in folders]
    nominal_v = 2.4
    for i, f in enumerate(freqs):
        spec= np.load('npy_files/disp_plot/'+ str(f) + '/1.0_chunk_0_1024.npy')
        psum = np.sum(spec, axis=1)
        f_grid= np.load('npy_files/disp_plot/'+ str(f) + '/chunk_0_f_grid.npy')
        kr = 2*np.pi*(f_grid-f)/2.4
        threshold = 100
        peak_inds, peak_heights = find_peaks(abs(psum), height=25)
        print(peak_heights)
        peak_heights =peak_heights['peak_heights']
        print(len(peak_inds), len(peak_heights), len(kr[peak_inds]))
        #plt.plot(kr, abs(psum))
        #plt.show()
        #plt.scatter(kr[peak_inds], peak_heights, color='r')
        plt.scatter([f]*len(peak_heights), kr[peak_inds], s=10)
        #plt.show()
    plt.suptitle('Dispersion for estimated kr')
    plt.xlabel('Source frequency (Hz)')
    plt.ylabel('kr (1/m)')
    plt.show()

load_psd()
    
    

