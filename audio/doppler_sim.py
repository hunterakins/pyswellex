import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.autoregressive import load_fest

'''
Description:
Use velocity estimates and adiabatic mode theory to simulate the first 50 minutes of the swellex experiment for the shallow source

Date: 
8-10-2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

def get_t_grid(start_min, end_min):
    """
    get the time grid from [start_min, end_min) in seconds
    """
    return np.arange(start_min*60, end_min*60, 1/1500)

def get_v(start_min, end_min,ref_freq=385):
    N, delta_n = 1500, 750
    fest, err, amp = load_fest(ref_freq, N, delta_n)
    print(fest.shape, err.shape)
    i1 = int(start_min*60*1500/delta_n)
    i2 = int(end_min*60*1500/delta_n)
    fest = fest[:,i1:i2]
    err = err[:, i1:i2]
    best_inds = np.argmin(err, axis=0)
    print(best_inds.size, fest.shape)
    i = np.linspace(0, best_inds.size-1, best_inds.size, dtype=int)
    fest = fest[best_inds,i]
    v= (ref_freq-fest)*1500/ref_freq
    grid = np.linspace(start_min*60, end_min*60 - delta_n/1500, fest.size) 
    new_grid = get_t_grid(start_min, end_min)
    new_v = np.interp(new_grid, grid, v)
    return new_v


def get_r(r0, v):    
    r = r0 + np.cumsum(v)*1/1500
    return r

def get_bathymetry():
    """
    Just eyeball the bathymetry to get a profile
    2.5 minute incremements
    """
    spacing = 2.5*60
    t_domain = np.arange(0, 75*60 + spacing, spacing)
    b_vals = np.array([250, 260, 220, 216, 212.5, 212.5, 210, 209, 206, 200, 200, 200, 200, 200, 200, 195, 190, 185, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180]) 
    print(len(b_vals))
    print(len(t_domain))
    plt.plot(t_domain/60, b_vals)
    plt.show()
    
    
get_bathymetry()
v = get_v(0, 50)
r = get_r(8650, v)

    
    
    
    
