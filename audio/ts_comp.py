import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert, firwin,convolve, detrend
from scipy.ndimage import convolve1d
from swellex.audio.config import get_proj_root, get_full_ts_name
import sys
#import pickle 

'''
Description:
Narrowband filter the swellex data

Date: 4/8/2020

Author: Hunter Akins
'''



def kay_preproc_vec(s_n, fc, order, df,save=False, froot=None):
    """
    Input 
    s_n - np array
        each row is a sensor time series
    fc - float
        center frequency of narrowband filter
    order - int
        filter length
    df - float
        1/2 the bandwidth of the narrowband filter
    Output
    u_n - numpy array with same dims as s_n
        hilbert transform of narrowband filter output
    if save is True, the bandpassed filtered data is saved
    at froot
    """
    filt = firwin(order, [fc-df, fc+df], fs=1, pass_zero=False, window='hann')
    s_n = convolve1d(s_n, filt, mode='constant')
    u_n = hilbert(s_n)
    if save == True:
        print('Saving the band pass filtered time series for ', str(1500*fc))
        np.save(froot + str(int(1500*fc)) + '_' + str(order) + '_ts.npy', s_n)
    return u_n


def get_bw(freq):
    """
    Get total bandwidth for source frequency freq
    Use formula
    fs/cw + f*3*(1/1450-1/1800) derived in my notes
    also add a little leeway for potential error in f_center
    
    """
    B = freq/1500 + freq*3*(1/1450 - 1/1700)
    fc_err = freq*(.4/1500)
    B += fc_err
    return B

def get_alt_bw(freq):
    """ 
    8-10 
    I want to try one more with a smaller bandwidth
    so i'm cutting it in half
    """
    B = get_bw(freq)
    B /= 2
    return B

def get_order(freq):
    """
    Using my derived bandwidth estimations for each frequency, pick the filter
    order that get me maximum noise rejection
    """
    B = get_bw(freq)
    T = 1/B
    N = T*1500
    N = np.power(2, int(np.log2(N))+1)
    return N

def get_fc(freq, proj_string):
    """
    Get the center doppler shifted frequency
    So this can be made way more complicated...
    we're gonna just assume a velocity around
    2.4 (so this will only work on the first 35 mins of data or so)
    I account for possible errors by increasing the bandwidth
    I can refine this later once I perform the frequency estimations
    """
    if proj_string == 's5':
        fc = freq*(1 + 2.4/1500)
    if proj_string == 'arcmfp1':
        fc= freq*(1 - 2.5/1500)
    return fc
    
def get_nb_fname(freq, proj_string='s5'):    
    order = get_order(freq)
    B = get_bw(freq)
    df = B/2
    proj_root = get_proj_root(proj_string)
    fname= proj_root + '_'.join([str(freq), str(order), str(df)[:6]]) + '.npy'
    print(df, fname)
    return fname

def filter_x(freq,proj_string='s5'):
    """
    For each frequency in freqs, 
    take in the raw time series on the array
    and filter it

    Input 
        freqs - list of floats/ints    
            frequencies at which to perform the 
            phase estimation
        df - bandwidth of td filter
        swellex3 - Bool
            distinguish between the swellex s5 event 
            and the swellex-3 arcmfp event
    Output 
        None
        saves an array for each frequency in freqs
        to /home/fakins/data with the name
        freq_pest.npy 
        if save is True, the bandpass filtered
        data is saved at froot """ 
    order = get_order(freq)
    B = get_bw(freq)
    df = B/2
    fc = get_fc(freq, proj_string)
    print('order', order, 'fc', fc, 'df', df)
    filt = firwin(order, [fc-df, fc+df], fs=1500, pass_zero=False, window='hann')
    fr, fva = np.fft.fftfreq(order, 1/1500), np.fft.fft(filt)
    raw_data_name = get_full_ts_name(proj_string)
    print('Loading ts at ', raw_data_name)
    x = np.load(raw_data_name)
    print('Appling the filter')
    x_ts = convolve1d(x, filt, mode='constant')
    fname = get_nb_fname(freq, proj_string=proj_string)
    print('Saving the filtered output to ', fname)
    np.save(fname, x_ts)
    return

if __name__ == '__main__':
    freq = int(sys.argv[1])
    proj_string = sys.argv[2]
    #start_ind = int(sys.argv[2])
    #end_ind = int(sys.argv[3])
    filter_x(freq, proj_string=proj_string)
#    tscc_coh_sum(freqs)


