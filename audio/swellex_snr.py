import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve, firwin, stft
'''
Description:
Estimate snr vs frequency for swellex 96 data

Estimate snr as function of time for each frequency
on first sensor in the array

I only look at the section from 6.5 minutes to 40 minutes 

Date: 

Author: Hunter Akins
'''



"""
Pick sensor """
sensor_ind = 2
def save_snr(sensor_ind):

    """ Pick signal bands """
    freqs = [49, 64, 79, 94, 109, 112, 127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    """ Adjust for doppler shift """
    freqs = [x*(1 + 2.4/1500) for x in freqs]

    """ Pick noise bands """
    nfreqs = [47,62,77,92,107,125,143,161,179,214,248,296,351,401]
    """ Select another selection for comparison """
    nfreqs1 = [x+2 for x in freqs]

    """ FFT segment size. fs=1500Hz, so 2048 is about 1.5 seconds"""
    block_size = 2048

    """ Load in the first sensor (numpy array)"""
    sensor = np.load('npy_files/sensor' + str(sensor_ind) + '.npy') 
    sensor -= np.mean(sensor) # zero mean

    """ Run stft to get spectrogram, use 50 percent overlap 
    My parameter choices give frequency resolution of .18 Hz
    (I  zero pad to attain this)"""

    f,t,z = stft(sensor[:,0], fs=1500, window='hamming', nperseg=block_size, nfft=block_size*4, noverlap = (8*block_size//16), return_onesided=True)

    """ this gives me the indices for the closest bands 
        to the signal frequencies  """
    def get_finds(f, freqs):
        """
        Get indices for the closest frequency bin
        Input
        f - np 1d array of floats
            Frequency grid, evenly spaced
        freqs - list of floats
            Frequencies who's indices you want
        df - float
        """
        finds = np.zeros(len(freqs),dtype=int)
        df = f[1]-f[0]
        for i in range(len(freqs)):
            freq = freqs[i]
            ind_float = freq/df
            ind = int(round(ind_float))
            finds[i] = ind
        return finds

    finds = get_finds(f, freqs)
    ninds = get_finds(f, nfreqs)
    ninds1 = get_finds(f, nfreqs1)

    """ estimate power on each block """
    def get_pow(finds, stft_z):
        """
        Compute magnitude squared of the appropriate rows in the stft
        Input
        finds - numpy array of ints
            indices of the bands of interest
        stft_z - numpy 2d array of complex 128
            stft of data
        """
        power = np.zeros((len(finds), stft_z.shape[1]))
        for i in range(len(finds)):
            ind = finds[i]
            band = stft_z[ind,:]
            tmp = np.square(abs(band))
            power[i,:] =  tmp
        return power
            
    def get_snr(finds, ninds, stft_z):
        """
        Produce a time series of snr estimates for 
        the frequencies in freqs using the noise bands
        in nfreqs and the stft_z
        The choice of segment and overlap etc. for the
        stft determines the time-frequency resolution
        of the estimates
        Plan is to use power in the freqs bands and compare to 
        power in the closest noise bin. Then assume the noise
        power is the same in the signal bin and use the difference
        to estimate the signal power.
        Then take the ratio and maybe throw a log in there
        Input 
        finds - numpy array of ints
            indices of the freq. bands of interest
        ninds - numpy array of ints
            indices of the noise bands of interest
        stft_z - numpy 2d array of complex 128
            columns correspond to fft of a specific time
        Output 
        snr - numpy 2d array
        """
        num_times = stft_z.shape[1]
        snr_ests = np.zeros((finds.size, num_times))
        """ For each frequency """
        for i in range(len(finds)):
            ind = finds[i] 
            """ find closest noise ind """
            ind_diffs = np.array([abs(x - ind) for x in ninds])
            closest_nind = np.argmin(ind_diffs)
            nind = ninds[closest_nind]
            tpow = get_pow([ind], stft_z)
            npow = get_pow([nind], stft_z)
            spow_est = tpow - npow
            ratio = spow_est/npow
            """ cast negative values to -40 dB """
    #        ratio[ratio<=0] = 1e-4
            snr_ests[i,:] = 10*np.log10(ratio)
        return snr_ests

    snr = get_snr(finds, ninds, z)
    snr_comp = get_snr(ninds1, ninds, z)

    #for i in range(len(freqs)):
    ##    plt.title('SNR estimate for ' + '{:4.2f}'.format(freqs[i]) + ' Hz band')
    #    plt.title('SNR Estimate for ' + str(int(freqs[i])) + ' Hz band\ncompared to \'SNR\' estimates for the band at ' + str(int(freqs[i]) + 2) + ' Hz band')
    #    plt.plot(t/60, snr[i,:])
    #    plt.plot(t/60, snr_comp[i,:])
    #    plt.xlabel('Time (minutes)')
    #    plt.ylabel('SNR Estimate (dB)')
    #    plt.legend(['Signal band', 'Reference (noise) band'])
    #    plt.show()
    #


            
            
    np.save('npy_files/snr' + str(sensor_ind) + '.npy', snr)
    np.save('npy_files/snr_freqs.npy',np.array(freqs))

for ind in range(2, 22):
    save_snr(ind)
        


