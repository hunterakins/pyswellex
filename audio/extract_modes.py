import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.tscc_demod_lib import make_fname, DemodDat
import pickle
import os 
from scipy.signal import find_peaks
'''

Description:
Use my spectral estimates to extract the modes
Gamma appears as a tuning parameter
Goal is to search over gamma and find the most promising peaks
and their associated projections along the array

Date: 
08/05/2020

Author: Hunter Akins


'''

""" This is a global parameter determined by fest_demod.py"""
gammas = np.arange(0.9, 1.1, 0.0005)
depths = np.linspace(94.125, 212.25, 64)
depths = np.delete(depths, 21)

def get_est(freq, gamma, load_f):
    """
    Input
    freq - int
        source freq
    gamma - float
        gamma parameter...
    load_f - function
        function that takes in freq and gamma and gets the estimates 
    """
    est = load_f(freq, gamma)
    return est

def get_pow(est):
    return np.sum(abs(est), axis=0)

def get_frs(freq, f_grid):
    krs = np.load('npy_files/' + str(freq) + '_krs.npy')
    krs = krs.real[:11]
    v_est = -(freq - np.mean(f_grid)) / freq * 1500
    frs = (krs-np.mean(krs)) * v_est / 2 / np.pi + np.mean(f_grid)
    return frs

def plot_db_spec(freq,gamma, chunk_list):
    trunc_lims = slice(100, -100)
    num_chunks = len(chunk_list)
    fig, axes = plt.subplots(num_chunks, 1, sharex=True)
    plt.suptitle('Gamma ' + str(gamma)[:6])
    for i in range(num_chunks):
        chunk_id = str(i)
        x = chunk_load(freq, gamma, chunk_id)
        est=  x.vals
        no_x = chunk_load(freq, 0, chunk_id)
        no_f = no_x.get_fgrid()
        no_est = no_x.vals
        f = x.get_fgrid()
        est_pow = get_pow(est)
        no_pow = get_pow(no_est)

        norm = np.max(est_pow)
        est_db = est_pow / norm
        est_db = 10*np.log10(est_db)
        no_db = no_pow / norm
        no_db = 10*np.log10(no_db)

        axes[i].plot(no_f[trunc_lims],no_db[trunc_lims], alpha=0.5)
        axes[i].plot(f[trunc_lims], est_db[trunc_lims], color='g')

        frs = get_frs(freq, f)

        for fr in frs:
            axes[i].set_ylim([-6, 0])
            axes[i].plot([fr]*10, np.linspace(-15, 0, 10), '-', alpha=.5, color='k')
            axes[i].set_title(str(x.start_min) + ' to ' + str(x.end_min))
        label = 'With phase '  
    axes[0].legend(['No phase', label],prop={'size':8})
    axes[-1].set_xlabel('Frequency (Hz)')
        #plt.legend(['Using 385 reference', 'Using 109 reference', 'Using no reference', 'Simulated kr, shifted using an assumed sound speed of 1530 m/s'])
    return fig

def chunk_load(freq, gamma,chunk_id, suffix=''):
    """ Load a single demod dat for source frequency freq, gamma val, and chunk_id """
    fname = make_fname('npy_files/' + str(freq) + '/chunk_exp/', gamma, chunk_id, suffix)
    with open(fname, 'rb') as f:
        x = pickle.load(f)
    return x


def animate_figs(freq):
    vid_name = str(freq) + '.mp4'
    os.chdir('pics/' + str(freq) + '/chunkexp')
    os.system('ffmpeg -r 10 -f image2 -s 1920x1080 -i %03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ' + vid_name)
    os.chdir('../../../')

def local_ascent(pow_vals, start_ind):
    """
    Cursory local ascent function
    Doesn't check for edges
    Finds the most immediate local peak
    """
    start_val = pow_vals[start_ind]
    """ Check left """
    l_val = pow_vals[start_ind-1]
    if l_val > start_val:
        last_val = start_val
        last_ind = start_ind
        curr_val = l_val
        curr_ind = start_ind-1
        while curr_val > last_val:
            last_ind = curr_ind
            last_val = curr_val
            curr_ind = last_ind - 1
            curr_val = pow_vals[curr_ind]
        return last_ind

    """ If this falls through, it's right """
    r_val = pow_vals[start_ind+1]
    if r_val > start_val:
        last_val = start_val
        last_ind = start_ind
        curr_val = r_val
        curr_ind = start_ind+1
        while curr_val > last_val:
            last_ind = curr_ind
            last_val = curr_val
            curr_ind = last_ind + 1
            curr_val = pow_vals[curr_ind]
        return last_ind
    else:
        return start_ind
        
        
    

def algorithm_1(freq, chunk):
    """
    Try and get the best estimate of each kr
    for the specific chunk
    Basic idea is to start at gamma = .1 to get a decent 
    sense of where the peaks are
    Then as gamma moves around, track the peak and retain the value
    and mode shape it has at its maximum
    """
    gamma = gammas[np.argmin(np.array([abs(x-1) for x in gammas]))]
    dat = chunk_load(freq, gamma, str(chunk))
    f_grid = dat.get_fgrid()
    est = dat.vals
    power = get_pow(est)
    norm = np.max(power)
    power /= norm
    pow_db = 10*np.log10(power)
    power *= norm
    ref_power = power
    peak_inds, _ = find_peaks(pow_db, height=-6)
    peak_heights = power[peak_inds]
    best_gammas = [gamma]*len(peak_inds)
    best_inds = peak_inds
    for g in gammas:
        print(g)
        dat = chunk_load(freq, g, str(chunk))
        est = dat.vals
        f = dat.get_fgrid()
        power = get_pow(est)
        for i,ind in enumerate(peak_inds):
            new_ind = local_ascent(power, ind)
            if power[new_ind] > peak_heights[i]:
                peak_heights[i] = power[new_ind]
                best_gammas[i] = g
                best_inds[i] = new_ind
    
    for i,g in enumerate(best_gammas):
        dat = chunk_load(freq, g, str(chunk))
        est = dat.vals
        power = get_pow(est)
        ind = best_inds[i]
        mode_shape = est[:,ind]
        plt.plot(mode_shape.real, depths)
        plt.plot(mode_shape.imag, depths)
        plt.show()
            


        
                
        
    
    
    

def track_comparison(freq):
    """
    For the demodulated data for the recent chunks
    Plot power for each gamma in a big subplot
    """
    gammas = gammas[::5]

    #chunk_load(109, gamma, '0')
    chunk_list = [0,1, 2, 3, 4]
    for i, gamma in enumerate(gammas):
        fig = plot_db_spec(freq, gamma, chunk_list)
        #fig.set_size_inches(32, 18)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.savefig('pics/109/chunkexp/' + str(i).zfill(3), bbox_inches='tight',dpi=500)
        plt.close(fig)


if __name__ == '__main__':
    freq = 109
    #fig = plot_db_spec(freq, ref_f, ref_est, no_f, raw_est)
    #fig.savefig('hihunt.png', dpi=500)


    algorithm_1(freq, 1)
