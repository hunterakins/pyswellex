import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy.signal import firwin, convolve, stft, detrend, decimate, get_window

f0 = 49
dats = np.load('npy_files/sim_p.npy')
dats.shape
for i in range(21):
    # LOAD DATA
    sensor = dats[i,:]
    f_filter = f0
    low_cut = f_filter - .5
    high_cut = f_filter + .5
    order = 1024
    fs= 1500
    vals = sensor
    # STFT
    block_size = 1024
    f,t,z = stft(vals, fs=1500, window='hamming', nperseg=block_size, nfft=block_size*4, noverlap = block_size//2, return_onesided=True)
    diffs = [abs(f_filter - x) for x in f]
    f_ind = np.argmin(diffs)
    f[f_ind]
    ts = z[f_ind,:]
    np.save('npy_files/' +str(f0) + str(i+1) + '_sim.npy', ts)

tmp = np.load('npy_files/' +str(f0) + str(0+1) + '_sim.npy')
tmp.size
size = tmp.size
full_data = np.zeros((21, size), dtype=np.complex128)
for i in range(21):
    dats = np.load('npy_files/' +str(f0) + str(i+1) + '_sim.npy')
    full_data[i,:] = dats

np.save('npy_files/' + str(f0) + '_ts_interp_sim.npy', full_data)


sensor1 = full_data[0,:]
plt.plot(sensor1)
deltaT = 512/200 # time between fft windows
v = 2.4
deltaR = deltaT*v
krs = 2*np.pi*np.fft.fftfreq(sensor1.size, 1/deltaR)
fvals = np.fft.fft(sensor1)
plt.plot(krs, abs(fvals))

plt.plot(sensor1)
