import numpy as np
from matplotlib import pyplot as plt
from env.env.envs import factory
from pyat.pyat.readwrite import read_modes
from kay import undiff, diff_seq
from scipy.signal import detrend, firwin, convolve,stft

'''
Description:
Generate a time series of data from a 54 meter deep source with a velocity that looks like
2.4 + .2*cos(omega*t)
where omega is 2*pi*.2 


Date: 6/20

Author: Hunter Akins
'''


""" Get the modes """
freq =49
env_builder  =factory.create('swellex')
env = env_builder()
zs = 54
zr = np.array([94.125, 99.755, 105.38, 111.00, 116.62, 122.25, 127.88, 139.12, 144.74, 150.38, 155.99, 161.62, 167.26, 172.88, 178.49, 184.12, 189.76, 195.38, 200.99, 206.62, 212.25])
num_rcvrs = zr.size
dz =  5
zmax = 216.5
dr = 1
rmax = 10*1e3

folder = 'at_files/'
fname = 'swell'
env.add_source_params(freq, zs, zr)
env.add_field_params(dz, zmax, dr, rmax)
p, pos = env.run_model('kraken', folder, fname, zr_range_flag=False)
modes = read_modes(**{'fname':folder+fname+'.mod', 'freq':freq})
print(modes.phi.shape)
As = modes.phi[0,:]
phi = modes.phi[1:,:] # throw out source depth
krs = modes.k
print('average kr', np.mean(krs))
print('k', 2*np.pi*freq/1491.8)
print('ratio', np.mean(krs)/(2*np.pi*freq/1491.8))

"""
Form the range grid on the first sensor """
v0=2.4 # m/s
fs = 1500
dt = 1/fs
dr = v0*dt
#r = np.arange(1000, 2101, dr) #1 m spacing
""" To get approximately 1km of data, I need approximately 1000 / dr steps """
num_times = int(1000 / dr)
tmax = num_times*dt
t = np.arange(0, tmax, dt)
f_ship = .2 #.2 Hz oscillations
omega_ship = 2*np.pi*f_ship
r0 = 1000
r = r0 + v0*t + omega_ship*.2*np.sin(omega_ship*t) + np.sqrt(.1)*np.random.randn(t.size)
field = np.zeros(r.size, dtype=np.complex128)
for i in range(krs.size):
    term = As[i]*phi[0,i]*np.exp(complex(0,1)*krs[i]*r)/np.sqrt(krs[i]*r)
    field += term

"""
Look at effect of low pass filter on k domain phase
"""
yn, zn = diff_seq(field)
ddp = np.angle(zn) 
p0 = np.angle(field[0])
phase_est = undiff(p0, ddp, yn)
phase_est -= phase_est[0]
og_phase =phase_est
approx_phase = np.mean(krs)*r
approx_phase -= approx_phase[0]
plt.figure()
plt.suptitle('Compare approximate phase and actual phase over 1km')
plt.plot(og_phase)
plt.plot(approx_phase)
plt.figure()
plt.suptitle('mean(kr)*r - true phase over a 1km track')
plt.plot(approx_phase-og_phase)

""" Add time dependence """
tfield = field*np.exp(complex(0,1)*2*np.pi*freq*t)

""" Narrowband filter """
df = .5
filt = firwin(1024, [freq-df, freq+df], fs=fs, pass_zero=False, window='hann')
filt_field = convolve(tfield, filt, mode='same')
plt.figure()
plt.plot(filt_field.real)

""" Estimate the phase """
un = filt_field
yn, zn = diff_seq(un)
ddp = np.angle(zn) 
p0 = np.angle(un[0])
phase_est = undiff(p0, ddp, yn)
phase_est -= phase_est[0]
theor_phase = np.mean(krs)*r  + 2*np.pi*freq*t
theor_phase -= theor_phase[0]

plt.figure()
plt.suptitle('mean(kr)*r+omegat - actual phase')
plt.plot(theor_phase-phase_est)

plt.figure()
plt.plot(detrend(phase_est))
plt.show()

"""
Look at kr domain 
"""
block_size = 1024
f,t,z = stft(filt_field.real, fs=1500, window='hamming', nperseg=block_size, nfft=block_size*4, noverlap = block_size//2, return_onesided=True)
ind = np.argmin(np.array([abs(freq - x) for x in f]))
band = z[ind,:]
plt.plot(band)
plt.show()
kr_dom = np.fft.rfft(band.real)
dr = v0*dt
kr_freqs = np.fft.rfftfreq(band.real.size, dr)
plt.plot(kr_freqs, abs(kr_dom))
plt.show()


def dphidr(As, phi, krs, r):
    amp = np.zeros(r.size,dtype=np.complex128)
    numer = np.zeros(r.size, dtype=np.complex128)
    for i in range(krs.size):
        term1 = As[i]*phi[0,i]
        for j in range(i):
            term2 = As[j]*phi[0,j]
            numer += term1*term2*np.cos((krs[i]-krs[j])*r)
            amp += 0
    return numer

dphidr(As, phi, krs, r)

