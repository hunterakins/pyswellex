import sys
#import matplotlib
#matplotlib.use('Qt5Agg')
#from matplotlib import pyplot as plt
import numpy as np
import pickle
from math import ceil,floor
from scipy.stats import mode
from scipy.io import loadmat, wavfile
from sioread.sioread import SioStream, sioread
from scipy.interpolate import interp1d

"""
Routines for processing swellex dataset.
"""

def load_swellex_stream():
    fname = '/home/hunter/data/swellex/audio/J1312315.vla.21els.sio'
    stream = SioStream(fname)
    return stream

def fetch_buffer(stream, start_min, end_min, step=1):
    """
    Input:
    start_min - float
    end_min - float
    Output:
    buffer - numpy array of data
    """
    if start_min < 0 :
        raise ValueError("Start min must be > 0, but is ", start_min)
    if end_min > 75:
        raise ValueError("end_min must be <= 75", end_min)
    start_index = 1500*60*start_min
    last_index = 1500*60*end_min
    arg = slice(start_index, last_index, step)
    vals = stream[arg]
    return vals

def get_sensor_i(s, i):
    """ Input
        s - string
        i - int (sensor index)
        output
        x = N*1 numpy array
    """
    for j in range(15):
        x = fetch_buffer(s, 5*j, 5*(j+1))
        """ just look at a single sensor """
        x1 = x[:,i].T
        datalen = x1.size
        print(datalen)
        if j == 0:
            sensor1 = np.zeros((datalen*15,1))
            print(sensor1.size)
        sensor1[j*datalen: (j+1)*datalen,0] = x1
    return sensor1

def single_sensor_fft(sensor, N, overlap):
    datalen = sensor.size
    rem = datalen % N
    overlap_N = int(N * overlap)
    num_ffts = int((datalen - rem) / overlap_N)
    if overlap_N < rem:
        num_ffts += 1
    freqs = np.fft.fftfreq(N, 1/1500)
    vals = np.zeros((num_ffts,freqs.size), dtype=np.complex128)
    window = np.hamming(N)
    for i in range(num_ffts):
        dats = sensor[i*overlap_N:i*overlap_N + N].reshape(N)
        dats = dats*window
        tmp = np.fft.fft(dats)
        vals[i,:] = tmp
    return freqs, vals

def save_s5():
    """
    Go through the 3 full 64 element sio files from velella
    and stitch together into 1 massive npy file
    64 by x dimensions
    """
    """ 
    Load first relevant siofile
    S5 event starts at J131 23:15 and goes to J132 00:30
    Therefore I should throw out the first 13 minutes of the
    first sio file
    ALso throw out timer thing in last row and switch the order
    This corresponds to 13*60*1500 samples
    """
    fname = '/home/fakins/data/Swellex/VLA_J131_2302'
    x,_ = sioread(**{'fname':fname})
    x = x[13*60*1500:,:64]
    x = x[:,::-1]
    x = x.T
    np.save('/home/fakins/data/part1.npy', x)
    """ Now get second relevant, we use all """
    fname = '/home/fakins/data/Swellex/VLA_J131_2332'
    x, _ = sioread(**{'fname':fname})
    x = x[:,:64]
    x = x[:,::-1]
    x = x.T
    np.save('/home/fakins/data/part2.npy', x)
    """ Now get third relevant. S5 ends at 0030, so throw out the last
    2 minutes of data, or2*60*1500 samples """
    fname = '/home/fakins/data/Swellex/VLA_J132_0002'
    x, _ = sioread(**{'fname':fname})
    x = x[:-2*60*1500,:64]
    x = x[:,::-1]
    x = x.T
    np.save('/home/fakins/data/part3.npy', x)

def stick_together():
    x1 = np.load('/home/fakins/data/part1.npy')
    x2 = np.load('/home/fakins/data/part2.npy')
    x3 = np.load('/home/fakins/data/part3.npy')
    num_samps = x1.shape[1] + x2.shape[1] + x3.shape[1]
    x = np.zeros((x1.shape[0], num_samps), dtype=np.dtype(x1[0,0]))
    print('x dimensions', x.shape)
    print('expected len', 75*60*1500)
    x[:,:x1.shape[1]] = x1[:,:]
    x[:,x1.shape[1]:(x1.shape[1]+x2.shape[1])] = x2[:,:]
    x[:,-x3.shape[1]:] = x3[:,:]
    np.save('/home/fakins/data/s5.npy',x)
    
    
    
    

if __name__ == '__main__':
    """ load stream """
    save_s5()
    stick_together()
    #fname = '/media/hunter/ExtHard/Data/swellex/VLA_J131_2302'
    ##s = sioread(**{'fname':fname})
    ##s = load_swellex_stream()
    #s = SioStream(fname)
    #x = fetch_buffer(s,0, 1)
    #print(x.shape)
    #plt.plot(x[:,0])
    #plt.show()
    #
    #y = np.arange(40, 75-1/1500/60, 1/1500/60)
    #sys.exit(0)
    #""" get full record on sensor 1 minutes """
    #i = 0
    #for i in range(21):
    #    sensor1 = get_sensor_i(s, i)
#   # dec_sensor = sensor1[::10]
    #    sensor1 = sensor1 - np.mean(sensor1)
    #    sensor1 = sensor1[int(40*60*1500):75*60*1500]
    #    np.save('npy_files/sensor' + str(i+1) + 'cpa.npy', sensor1)
    #    print(sensor1.size)


    ##freqs, vals = single_sensor_fft(short_sens, 2048, 1)
    ##freqs = freqs[:freqs.size//2]
    ##vals = vals[:,:freqs.size//2]
    ##print(freqs[174])
    ##samp1 = vals[:,175]
    ##samp2 = vals[:,174]
    ##samp3 = vals[:,173]
    ##T = 1024/1500
    ##dom = np.linspace(0, samp1.size*T, samp1.size) *3
    ##amp = .02
    ##plt.plot(dom, amp*np.sin(2*np.pi*dom/15))
    ##plt.plot(dom, samp1.real)
    ##plt.plot(dom, samp2.real)
    ##plt.plot(dom, samp3.imag)
    ##plt.show()
#
    ##spec = np.fft.fft(samp2)
    ##max_spec = np.max(abs(spec))
    ##pow = abs(spec)/max_spec
    ##plt.plot(np.log10(pow))
    ##plt.show()
