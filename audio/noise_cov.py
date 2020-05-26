import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import firwin, convolve

'''
Description:
Look at covariance of noise along array, as well as autocorrelation function of noise on single arrays
Idea is to test the whiteness assumption in both the time sense and the spatial sense.

Another thing to investigate is stationarity of the noise? Doesn't really matter much since I can 
assume gaussian for the whole thing but just use a moving sample covariance average

I need to compare the covariance of my residuals. It will be a good way to see if what's
left over is noise or signal. 

Also understanding the relationship between my bandpass filter and the correlation of the noise.
I need to basically simulate noise and run it through my processing and compare before
and after.


Date: 
5/18

Author: Hunter Akins
'''




'''
Description:
Look at noise correlation structure on the array

Date: 
5/10

Author: Hunter Akins
'''


def get_sample_chunk(nfreq, chunk_len, start_time=0, sensor_ind=1):
    """
    Take in a noise frequency and a chunk length in seconds
    Go ahead and snag the first chunk_len  of sensor1 and 
    filter around nfreq
    """

    """ Load sensor """
    sensor = np.load('npy_files/sensor' + str(sensor_ind) + '.npy')
    sensor = sensor.reshape(sensor.size)
    start_ind = int(start_time*1500)
    end_ind = int(chunk_len*1500)+start_ind
    sensor = sensor[start_ind:end_ind]

    """ Grab the 125 hz noise band """
    filt_order = 2048
    df = .5
    filt = firwin(filt_order,[nfreq-df, nfreq+df], fs=1500, pass_zero=False, window='hann')
    noise = convolve(sensor, filt, mode='same')
    return noise

def comp_noise_simple(freq, nfreq, chunk_len, start_time=0,sensor_ind=1):
    plt.figure()
    plt.subplot(211)
    noise = get_sample_chunk(nfreq, chunk_len,start_time,sensor_ind)
    plt.plot(noise)
    plt.subplot(212)
    sig = get_sample_chunk(freq, chunk_len,start_time, sensor_ind)
    plt.plot(sig)

def filter_wgn(time_len,freq):
    """ 
    Generate a wgn sequence of length 1500*time_len
    Bandpassfilter and look at it 
    """
    num_samps = int(time_len*1500)
    wgn = np.random.randn(num_samps)
    filt_order = 1024
    df = .5
    filt = firwin(filt_order,[freq-df, freq+df], fs=1500, pass_zero=False, window='hann')
    noise = convolve(wgn, filt, mode='same')
    return noise

def plot_filter_transfer(freq):
    filt_order = 2048
    df = .5
    filt = firwin(filt_order,[freq-df, freq+df], fs=1500, pass_zero=False, window='hann')
    freqs = np.fft.rfftfreq(filt_order, d = 1/1500)
    vals = np.fft.rfft(filt)
    plt.plot(freqs, vals)

def get_chunk_on_array(freq, chunk_len, start_time=0):
    """
    Get the measurements in the time time interval
    [start_time, start_time + chunk_len]
    on the full array at frequency freq
    """
    for i in range(21):
        sensor_ind = i + 1
        chunk = get_sample_chunk(freq, chunk_len, start_time, sensor_ind)
        if i == 0:
            array_vals = np.zeros((21, chunk.size))
        array_vals[i,:] = chunk
    return array_vals
        
    


if __name__ == '__main__':
    freqs = [49, 64, 79, 94, 109, 112, 127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    time_sec = 60
    nfreqs = [47,62,77,92,107,125,143,161,179,214,248,296,351,401]
    nfreq = 125
    narr = get_chunk_on_array(nfreq, 10, 0)
    for i in range(5):
        plt.plot(narr[i,:])
        plt.show()





