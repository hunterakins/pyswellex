import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import spectrogram, detrend

'''
Description:
Compute spectrogram of raw time series on first element of array time series

Date: 
8/28/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


def save_arc_spec():
    x = np.load('/home/fakins/data/arcmfp1.npy')
    print(x.shape)
    x0 = x[0,:]
    print(x0.shape)

    N = np.power(2, int(np.log2(10*1500))+1) # over ten second fft
    N_overlap = N//2

    f, t, sxx = spectrogram(x0, fs=1500, nperseg=N, noverlap=N_overlap, detrend=detrend) 
    np.save('/oasis/tscc/scratch/fakins/data/arcmfp1/f.npy', f)
    np.save('/oasis/tscc/scratch/fakins/data/arcmfp1/t.npy', t)
    np.save('/oasis/tscc/scratch/fakins/data/arcmfp1/sxx.npy', sxx)
    fig = plt.figure()
    plt.contourf(t, f, sxx)
    plt.savefig('spectro.png')
    plt.close(fig)


def plot_arc_local():
    f = np.load('npy_files/f.npy')
    t = np.load('npy_files/t.npy')
    sxx = np.load('npy_files/sxx.npy')
    sxx = sxx[:600, :]
    sxx /= np.max(sxx, axis=0)
    sxx_db = np.log10(sxx / np.max(sxx))
    print(f.shape, t.shape, sxx.shape)
    plt.contourf(t, f[:600], sxx)
    plt.show()

if __name__ == '__main__':
    #plot_arc_local()
    save_arc_spec()
