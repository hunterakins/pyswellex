import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy.signal import firwin, convolve, stft, detrend, decimate, stft
from swellex.ship.ship import load_track, get_velocity
import os

# it's a 5.1 km ship track

f0 = 49
dats = np.load('npy_files/' + str(f0)+'_ts_interp.npy')
norm_dats = dats / np.sqrt(2*np.var(dats))
phases = np.load('npy_files/' + str(f0) + 'phase_estimates.npy')
time_per_sample =512/1500
v = 2.4
range_per_sample = v*time_per_sample
times = np.linspace(0, dats.shape[1]*time_per_sample, dats.shape[1])

phases.size
dats.size
nco_i = np.cos(2*np.pi*f0*times + phases)
plt.plot(times, dats)
plt.plot(times/60, phases)
range_km, lat, lon, time = load_track()
plt.plot(time, range_km)
velocity = get_velocity(time, range_km)
plt.plot(time, velocity)

fvals = np.fft.fft(dats[0,:])

plt.plot(np.square(abs(fvals)))
block_size = 1024
f,t,z = stft(dats, fs=1500, window='hamming', nperseg=block_size, noverlap = block_size//2)
f[33]
ts = z[33,:]
plt.plot(t/60, ts.imag)
krs = 2*np.pi*np.fft.fftfreq(times.size, 1/range_per_sample)
plt.plot(krs, abs(np.fft.fft(ts)))

"""
Low pass filter the phase phase_estimates
"""
order = 1024
f_cutoff = 1 / 12
filter = firwin(order, f_cutoff, fs=1500)
filter.shape
lpphase = convolve(phases, filter, mode='same')
plt.plot(lpphase)
plt.plot(phases)
plt.plot(phases-lpphase)


"""
Compare ot yang plots
"""
indices = slice(int(4.5*1500*60), 8*1500*60,1)
vals = phases[indices]
lpvals = lpphase[indices]
times = np.linspace(4.5, 8, vals.size)
plt.plot(times, vals)
plt.plot(times, lpvals)
