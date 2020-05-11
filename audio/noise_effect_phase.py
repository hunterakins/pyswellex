import numpy as np
from matplotlib import pyplot as plt
from kay import get_p_est
from scipy.signal import detrend

'''
Description:
Look at the effect of white gaussian noise on on phase estimation.
Maybe look at effect of correlated noise afterwards.

Date: 

Author: Hunter Akins
'''

def make_data(snr):
    """
    Generate a narrowband time series corrupted with noise
    Signal will have amplitude 1, so signal variance will
    be 1/2
    For fn = 1, fs = 1500, fn = f*delta_t = f / fs
    This determines the noise variance 
    """

    """ Generate a sinusoid """
    t = np.arange(0, 101, 1/1500)
    fn = .25
    f = fn*1500
    df = .5
    dfn = .5/1500
    print(f)
    vals = np.cos(2*np.pi*f*t)

    """ Corrupt it with noise """ 
    n_var = 1 / 2 / np.power(10, snr/10)
    noise_vals = np.sqrt(n_var) * np.random.randn(vals.size)
    print(n_var)
    corrupt_vals = vals + noise_vals
    p_est = get_p_est(corrupt_vals, fn,dfn, 1024)
    return t,p_est


snrs = [-30, -25, -20, -15]#, 10, 20, 30,]
for x in snrs:
    t,p_est = make_data(x)
    plt.figure()
    plt.xlabel("Time (s)")
    plt.plot(t, detrend(p_est))
plt.show()




