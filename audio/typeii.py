import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import firwin, convolve

'''
Description:
Implement a type 2 phase frequency detector

Date: 
3/26/2020

Author: Hunter Akins
'''


""" Phase detector """

f = 49
fs = 1500
T = 20#200*1/49 # ten cycles
dt = 1/fs
times = np.linspace(0, T, T*fs)
tracksig = np.sin(2*np.pi*f*times)
zeros = np.zeros(tracksig.size)
tracksig = tracksig >= zeros # get it into square wave

vco_phs = .5
vco_f = 45
lsig = 0
lvco = 0

qsig, qvco = 0, 0 # flip flop output
lsig, lvco = 0, 0 # keep track of last for edge detection
leads = []
lags = []
vco_vals = []
rsts = []

"""
Selecting filter coefficients (Horowitz and Hill pg. 963)
"""
f2 = 1 # 1 Hz unity gain frequency
fzero = 5
order=35
filt = np.ones(order)/order
filt = firwin(order, f2, fs=fs)
errs = order*[0]
ferrs = []
vco_fs = []
Kp = .01
Kf =.001

for i in range(times.size):
    vco_phs = (vco_phs + 2*np.pi*vco_f*dt)%(2*np.pi)
    vco = (vco_phs < np.pi)
    vco_vals.append(vco)
    sig = tracksig[i]
    """
    Detect when both flip-flop outputs are high
    """
    rst = not(qsig & qvco) 
    rsts.append(rst)
    """
    Detect leading edges of sigal and vco 
    """
    sig_trigger = (not lsig) & sig #signal low to high
    vco_trigger = (not lvco) & vco
    """ 
    Look at flip flop output to determine lead and lag values 
    if trigger conditions are met
    If the trigger condition is met, output should be high unless rst got hit
    If trigger condition is not met, output should stay the same unless rst got hit
    """
    """
    qsig was high last time and qvco is still low...
    or it just got triggered high 
    """
    qsig = (sig_trigger or qsig)&rst
    """
    qvco was high on the last thing and qsig is still low
    """
    qvco = (vco_trigger or qvco)&rst
    errsig = int(qsig) - int(qvco)
    errs.append(Kp*errsig)
    lags.append(qsig)
    leads.append(qvco)
    lsig = sig
    lvco = vco

    """
    Use PFD Output to steer vco in right direction
    """
    filt_err = np.sum(convolve(errs[-order:], filt, mode='same'))
    ferrs.append(filt_err)
    vco_fs.append(vco_f)
    vco_f += Kp*errsig + Kf*filt_err
    
    
    
    
    
plt.subplot(511) 
plt.plot(tracksig)
plt.subplot(512)
plt.plot(vco_vals)
plt.subplot(513) 
plt.plot(errs)
plt.subplot(514)
plt.plot(ferrs)
plt.subplot(515)
plt.plot(times, vco_fs)
plt.show()
