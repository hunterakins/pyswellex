import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import firwin, convolve

'''
Description:
Try a simple polynomial model for ship motion. 
Result is a polynomial model for phase.
Perform MLE on 1 second chunks of data. 
See if I get consistent velocity estimates.

Date: 
3/24/2020

Author: Hunter Akins
'''

def bp_filter_dats(f0, sensor, order=512):
    f_filter = f0*(1 + 2.4/1500) # from Yang pf. 1681 after equation 7
    low_cut = f_filter - .5
    high_cut = f_filter + .5
    fs = 1500
    filter_vals = firwin(order, [low_cut, high_cut], fs=fs, pass_zero=False, window='hann')
    vals = convolve(sensor[:,0], filter_vals, mode='same')
    return vals

def get_replica(f0, delta, alpha, times):
    replica = np.cos(2*np.pi*f0*times + delta + alpha*times)
    return replica

def get_sin_replica(f0, delta, alpha, times):
    replica = np.sin(2*np.pi*f0*times + delta + alpha*times)
    return replica
    

def J(data, f0, delta, alpha, times):
    replica = get_replica(f0, delta, alpha, times)
    diff= data-replica
    return np.sum(np.square(diff))

def dJdAlpha(data, f0, delta, alpha, times):
    cos_term = get_replica(f0, delta, alpha, times)
    sin_term = get_sin_replica(f0, delta, alpha, times)
    diffs = data-cos_term
    data = diffs*times
    data = data*sin_term 
    deltaJ = 2*np.sum(data)
    return deltaJ

def dJdDelta(data, f0, delta, alpha, times):
    cos_term = get_replica(f0, delta, alpha, times)
    sin_term = get_sin_replica(f0, delta, alpha, times)
    diffs = data-cos_term
    data = data*sin_term 
    deltaJ = 2*np.sum(data)
    return deltaJ
    


sensor = np.load('npy_files/sensor1.npy')

"""
Try on one second first, at 49 Hz
"""
f0 = 49
T = 60*15
N = 1500*T
vals = bp_filter_dats(f0, sensor[:N], order=4096)
vals =vals[:N]
vals = vals/np.sqrt(np.var(vals)) 


times = np.linspace(0, T, N)

"""
PLL
"""
fg = f0
phig = 0
err_log = []
phi_log = []
err_sum = 0
K1, K2 = .001, .0001
quad_func = lambda t, phi: -np.sin(2*np.pi*fg*t + phi)
for i in range(vals.size): 
    phi_log.append(phig)
    vco_val = quad_func(times[i], phig)
    phi_err = vco_val*vals[i]
    err_log.append(phi_err)
    err_sum += phi_err
    phig = phig + K1*phi_err + K2*err_sum

plt.figure()
plt.plot(phi_log)

plt.figure()
plt.plot(err_log)

#plt.figure()
#plt.plot(np.cos(2*np.pi*fg*times + phi_log))
#plt.plot(vals)
plt.show()

