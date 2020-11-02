import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import pickle
from swellex.audio.config import get_proj_tones
from swellex.audio.lse import get_doppler_est_pickle_name, AlphaEst

'''
Description:
Routines to look at the phase estimates
for the various projects

Date: 
10/21

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


def get_local_pickle_dir(proj_str):
    proj_root = 'pickles/' + proj_str + '/'
    return proj_root

def copy_pests(proj_str):
    freqs = get_proj_tones(proj_str)
    for freq in freqs:
        pickle_loc = get_doppler_est_pickle_name(freq, proj_string=proj_str)
        ssh_root = 'fakins@tscc-login.sdsc.edu:'
        local_spot = get_local_pickle_dir(proj_str)
        os.system('scp ' + ssh_root + pickle_loc + ' ' + local_spot)

def get_local_name(freq, proj_str):
    pickle_loc = get_doppler_est_pickle_name(freq, proj_string=proj_str)
    i = len(pickle_loc)-1
    while pickle_loc[i] != '/':
        i -= 1
    i += 1
    pname = pickle_loc[i:]
    pdir = get_local_pickle_dir(proj_str)
    return pdir + pname
        

def load_pest(freq, proj_str, chunk_len=1024, var=0.1):
    name = get_local_name(freq, proj_str)
    with open(name, 'rb') as f:
        alpha_est = pickle.load(f) 
        freqs = np.array(alpha_est.freqs)
    freqs = freqs.reshape(freqs.size)
    num_samps = freqs.size
    t_grid = np.linspace(chunk_len/1500* 0.5, chunk_len/1500 * (num_samps-1) + chunk_len / 1500*0.5,num_samps) 
    return t_grid, freqs


def get_full_freqs(proj_str):
    c = 1500
    tmax = 33.5*60
    freqs = get_proj_tones(proj_str)
    first = True
    for count, f in enumerate(freqs):
        t_grid, f_ests = load_pest(f, proj_str)
        if first == True:
            i = 0 
            while t_grid[i] < tmax:
                i += 1
            first = False
            full_freqs = np.zeros((len(freqs), i-1))
        v_est = (f_ests/f - 1)*c
        full_freqs[count, :] = v_est[1:i]

    return t_grid[1:i], full_freqs

proj_str = 's5_deep'
#copy_pests(proj_str)
t3, ff3 = get_full_freqs(proj_str)
proj_str = 's5_shallow'
#copy_pests(proj_str)
t4, ff4 = get_full_freqs(proj_str)
proj_str = 'arcmfp1'
t1, ff1 = get_full_freqs(proj_str)
proj_str = 'arcmfp1_source15'
t2, ff2 = get_full_freqs(proj_str)


def make_theta_plots(tgrid, full_vest, freq, proj_str, c=1500):
    """
    Look at variability in phase correction
    as function of source frequency phase used 
    """
    mean_v = np.mean(full_vest, axis=1)
    dev_v = full_vest - mean_v.reshape(mean_v.size, 1)
    dev_f = freq*dev_v/c
    dt = 1024/1500
    theta = np.cumsum(dev_f,axis=1)*2*np.pi*dt
    print(theta.shape)
    fig, axes = plt.subplots(2,1)
    plt.suptitle('Excess phase accumulated due to accelerations for experiment ' + proj_str + ' estimated using the phase from each frequency band in the experiment')
    for i in range(theta.shape[0]):
        axes[0].plot(tgrid/60, theta[i,:] / 2 /np.pi)
        axes[1].plot(tgrid/60, dev_f[i,:])
    axes[1].set_xlabel('Time (min)')
    axes[0].set_ylabel('Excess phase (cycles)')
    axes[1].set_ylabel('Frequency deviation from mean (Hz)')

def make_range_err(tgrid, full_vest, freq, proj_str, c=1500):
    """
    Look at variability in phase correction
    as function of source frequency phase used 
    """
    mean_v = np.mean(full_vest, axis=1)
    dev_v = full_vest - mean_v.reshape(mean_v.size, 1)
    dt = tgrid[1]-tgrid[0]
    dev_r = np.cumsum(dev_v, axis=1)*dt
    fig = plt.figure()
    plt.suptitle('Range error in assuming a constant velocity')
    for i in range(dev_r.shape[0]):
        plt.plot(tgrid/60, dev_r[i,:])
    plt.ylabel('Excess range (m)')


make_range_err(t1, ff1, 53, 'arcmfp1')
plt.legend([str(f) for f in get_proj_tones('arcmfp1')])
plt.show()

make_theta_plots(t1, ff1, 85, 'arcmfp1')
plt.legend([str(f) for f in get_proj_tones('arcmfp1')])
make_theta_plots(t2, ff2, 85, 'arcmfp1_source15')
plt.legend([str(f) for f in get_proj_tones('arcmfp1_source15')])
make_theta_plots(t3, ff3, 64, 's5_deep')
plt.legend([str(f) for f in get_proj_tones('s5_deep')])
make_theta_plots(t4, ff4, 109, 's5_shallow')
plt.legend([str(f) for f in get_proj_tones('s5_shallow')])
plt.show()
#copy_pests(proj_str)



    
