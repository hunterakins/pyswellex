import numpy as np
from matplotlib import pyplot as plt
from swellex.ship.ship import get_good_ests, good_time
from swellex.audio.autoregressive import load_fest, esprit
from swellex.audio.ts_comp import get_fname, get_bw, get_order
from scipy.signal import detrend, firwin, convolve, lombscargle, find_peaks, hilbert, cheby1, sosfilt, decimate
from scipy.interpolate import interp1d
import scipy.linalg as la
import pickle
import multiprocessing as mp
import os
import sys
import json
import time

'''
Description:
Use tscc to demodulate data 

Date: 6/23/2020

Author: Hunter Akins
'''

def load_ship_ests(source_depth, chunk_len, ship_root = '/oasis/tscc/scratch/fakins/', r0=7703):
    if source_depth == 9:
        v_ests =np.load(ship_root + str(chunk_len) + 'sw_rr.npy')
        v_ests = v_ests[:-4]
        dt = chunk_len/1500
        times = np.linspace(0, (v_ests.size-1)*dt, v_ests.size)
        i = np.linspace(0, times.size-1, times.size, dtype=int)
    else:
        v_ests =np.load(ship_root + str(chunk_len) + 'dw_rr.npy')
        i, times, v_ests = get_good_ests(chunk_len, v_ests)
    r = [900]
    cpa_ind = np.argmin(abs(times-(59.8*60)))
    counter =  0
    curr_ind = cpa_ind
    while curr_ind < len(v_ests):
        dt = times[curr_ind]-times[curr_ind-1]
        v = v_ests[curr_ind]
        dr = dt*v
        r.append(r[-1] + dr)
        curr_ind += 1
    curr_ind = cpa_ind-1
    while curr_ind >= 1:
        dt = times[curr_ind-1]-times[curr_ind]
        v = v_ests[curr_ind]
        dr = dt*v
        r = [r[0] + dr] + r
        curr_ind -= 1
    print(len(times), len(r), len(v_ests))
    return i, times, r, v_ests

def get_phase_factors(freq, chunk_len, ship_root, source_depth, bias=1):
    """
    Doppler shift offsets the phase of each chunk
    Input 
    freq - int or float
        source freq
    chunk_len - int
        num samples in chunk
    ship_root - string
        location of ship velocity estimtaes
    source_depth - int
        9 or 54
    Output - 
    phase_factors - np array
        multiply your demod by this
    """

    i, t, r, v =load_ship_ests(54, chunk_len, ship_root)
    dt = t[1:] - t[:-1]
    phase_offset = [0]
    for delta_t, v_est in zip(dt, v[:-1]):
        delta_fs = 49*(- v_est/1500)*bias
        phase_extra = 2*np.pi* delta_fs*delta_t
        phase_offset.append(phase_offset[-1] + phase_extra)
    phase_offset = np.array(phase_offset)
    phase_factor = np.exp(-complex(0,1)*phase_offset)
    return phase_factor

def load_rel_data(freq, good_indices, r, chunk_len, v_ests, t):
    """
    Load up the relevant sensor time series
    Use the velocity estimates to complex baseband it
    Input 
    freq - int or float
        Source frequency
    good_indices - list
        The good chunks to keep
    r - np array
        Ranges
    chunk_len - int
        number of samples in a chunk
    v_ests - np array
        velocity estimates to go with the chunks
    t - np array
         time of start of good chunks
    """
    x = np.load('/oasis/tscc/scratch/fakins/' + str(freq) + '_ts.npy')
    #x = np.load('49_mini.npy')
    """
    Complex baseband it 
    """
    num_chunks = len(good_indices)
    vals = np.zeros((x.shape[0], num_chunks), dtype=np.complex128)
    counter = 0
    dt = 1/1500
    t_dom = np.linspace(0, chunk_len/1500 - dt, chunk_len)
    for i in good_indices:
        v = v_ests[counter] 
        f = freq*(1 - v/1500)        
        lo = np.exp(complex(0, 1)*2*np.pi*f*t_dom)
        lo = lo.reshape(lo.size, 1)
        rel_samp = x[:,i*chunk_len:(i+1)*chunk_len]
        val = rel_samp@lo
        vals[:,counter] = val[:,0]
        counter += 1
    return vals

def get_rel_vests(x, v_ests, chunk_len, start_min=0, end_min=40):
    """ load up the velocity estimates for the dat portion
    from 6.5 mins to 35 mins 
    return the interpolated stuff too"""
    T = x.size / 1500
    t = np.linspace(0, (v_ests.size-1)*chunk_len/1500, v_ests.size)
    dt = 1/1500
    t_grid = np.arange(start_min*60, end_min*60, dt)
    v_func = interp1d(t, v_ests)
    rel_v = v_func(t_grid)
    return rel_v

def get_Theta(df, theta0=0):
    """
    Compute the integrated phase deviation from the mean frequency 
    computed by integrating the deviation df (Hz)
    Input 
    df - np array
        deviation from mean frequency [f(t) - mean_f ]
    theta0 - float (int is fine too) 
        initial phase offset to start at 
        doesn't actually matter if you're doing incoh"""
    dt = 1/1500
    Theta = np.cumsum(df[::-1])*2*np.pi*dt
    Theta= -Theta[::-1]
    return np.array(Theta)

def get_F(gamma, Theta):    
    """
    For selected value of gamma (ratio of k_m / bar{k} )
    get phase drift removal factor from Theta
    Input 
    gamma - float
    Theta - np array
    """
    return np.exp(-complex(0,1)*gamma*Theta)

def f_search(f_grid, x, start_min=0, end_min=40):
    """
    Search through grid given by params
    Compute DFT at each of those frequencies
    input
    f_min - float
    f_max - float
    df - float
        freq spacing ALL IN HERTZ 
    x - np ndarray
        Rows are time series corresponding to sensor
    Out
    
    vals - np ndarray
        dims are num_sensors x num_freqs
        complex type
    """

    """ Convert to omega for convenience"""
    now = time.time()
    omegs = 2*np.pi*f_grid
    """ get relevant time grid """
    t = np.arange(start_min*60, end_min*60, 1/1500)
    print(t.size)
    mat = np.zeros((len(omegs), t.size), dtype=np.complex128)
    print(mat.shape, x.shape)
    i = 0
    for omeg in omegs:
        mat[i, :] = np.exp(-complex(0,1)*omeg*t)
        i += 1
    print('array population time', time.time()-now)
    now = time.time()
    vals = mat@x.T
    print('actual dft time', time.time()-now)
    return np.array(vals)

def get_f_mat(f_grid, start_min, end_min, deep=False):
    omegs = 2*np.pi*f_grid
    """ get relevant time grid """
    t = np.arange(start_min*60, end_min*60, 1/1500)
    omegs = omegs.reshape(omegs.size, 1)
    if deep == True:
        good_inds = [i for i in range(len(t)) if good_time(t[i])]
        t = t[good_inds]
    t = t.reshape(1, t.size)
    now = time.time()
    mat = np.exp(-complex(0,1)*omegs*t)
    print('second arr pop time', time.time() -now)
    now = time.time()
    mat = np.zeros((len(omegs), t.size), dtype=np.complex128)
    i = 0
    for omeg in omegs:
        mat[i, :] = np.exp(-complex(0,1)*omeg*t)
        i += 1
    print('array population time', time.time()-now)
    return mat

def gamma_iter(gamma, data, Theta,  mat):
    """
    For value of gamma, remove phase noise and
    estimate fourier transform for values in f_grid
    Input 
    gamma - float
        read the Walker pape...
    data - np ndarray dtype float64
        rows are sensors, columns are snapshots in time
    Theta - np array (1d)
        estimated phase noise ~present~ in the signal
    mat - np matrix 
        grid of frequencies to compute dft for
    Output 
    vals - np ndarray
        63 by f_grid.size, dtype = np.complex128
    """
    now = time.time()
    F = get_F(gamma, Theta) # this is the phase noise removal factor
    x = data*F
    vals = mat@x.T
    print('Mat mul time', time.time() - now)
    now = time.time()
    vals = vals.T
    print('Transpose time', time.time()-now)
    return vals

def save_demod(freq, gamma, deep=False,chunk_len=1024, start_min=0, end_min=40,suffix=''):
    """ Fetch data """
    if deep==True:
        v =np.load('/oasis/tscc/scratch/fakins/' + str(chunk_len) + 'dw_rr.npy')
        #v = np.load('../ship/' + str(chunk_len) + 'dw_rr.npy')
        v = v[:-4]
    else:
        i, t, r, v =load_ship_ests(9, chunk_len)
    x = np.load('/oasis/tscc/scratch/fakins/' + str(freq) + '_ts.npy')
    ind1 = int(start_min*60*1500)
    ind2 =int(end_min*60*1500)
    """ Narrow it down """
    x = x[:,ind1:ind2]
    t = np.arange(start_min*60, end_min*60, 1/1500)
    if deep == True:
        """ Apply a mask to get rid of the bad sections """
        mask = np.ones(x.shape)
        bad_inds = [i for i in range(len(t)) if not good_time(t[i])]
        print('bad_inds', bad_inds)
        mask[:, bad_inds[:]] = 0
        x = x*mask
    """ Interpolate v onto the relevant part of experiment """
    rel_v = get_rel_vests(x, v, chunk_len, start_min, end_min)
    f = freq*(1-rel_v/1500)
    mean_f = np.mean(f)
    dev_f = f- mean_f
    Theta = get_Theta(dev_f)
    print('gamma', gamma)
    Delta_f = .03 # 
    f_min = mean_f - Delta_f/2
    f_max = mean_f + 3*Delta_f/4
    df = .00005
    f_grid = np.arange(f_min, f_max, df) 
    np.save('/oasis/tscc/scratch/fakins/' + str(freq) + '/' + suffix + 'f_grid' + '.npy', f_grid)
    gamma_iter(gamma, x, Theta,f_grid)
    return
        
def mode_filter(freq, deep=False,chunk_len=1024, start_min=0, end_min=40,suffix=''):
    """ Fetch data """
    if deep==True:
        v =np.load('/oasis/tscc/scratch/fakins/' + str(chunk_len) + 'dw_rr.npy')
        #v = np.load('../ship/' + str(chunk_len) + 'dw_rr.npy')
        v = v[:-4]
    else:
        i, t, r, v =load_ship_ests(9, chunk_len)
    mode_mat= np.load('/home/fakins/code/swellex/audio/npy_files/' + 'mode_mat_' + str(freq) + '.npy')
    inv = np.linalg.pinv(mode_mat)
    x = inv@x
    ind1 = int(start_min*60*1500)
    ind2 =int(end_min*60*1500)
    """ Narrow it down """
    x = x[:,ind1:ind2]
    t = np.arange(start_min*60, end_min*60, 1/1500)
    if deep == True:
        """ Apply a mask to get rid of the bad sections """
        mask = np.ones(x.shape)
        bad_inds = [i for i in range(len(t)) if not good_time(t[i])]
        print('bad_inds', bad_inds)
        mask[:, bad_inds[:]] = 0
        x = x*mask
    """ Interpolate v onto the relevant part of experiment """
    rel_v = get_rel_vests(x, v, chunk_len, start_min, end_min)
    f = freq*(1-rel_v/1500)
    mean_f = np.mean(f)
    dev_f = f- mean_f
    Theta = get_Theta(dev_f)
    Delta_f = .03 # total frequency band size
    f_min = mean_f - Delta_f/2
    f_max = mean_f + 3*Delta_f/4
    df = .00005
    f_grid = np.arange(f_min, f_max, df) 
    np.save('/oasis/tscc/scratch/fakins/' + str(freq) + '/' + suffix + 'f_grid' + '.npy', f_grid)
    gamma_iter(0, x, Theta,f_grid, chunk_len, freq, suffix=suffix, start_min=start_min, end_min=end_min)
    return

def doppler_sim():
    """ 
    Use my data-derived velocity estimates
    to simulate the received field on the array
    Using Walker equatoin 15 for the simulation with
    chi_m = 1 
    """
    
    freq = 127
    chunk_len = 1024

    """ get the ship estimates """
    i, t, r, v =load_ship_ests(9, chunk_len,ship_root='../ship/')
    start_min, end_min = 5, 30
    ind1, ind2 = start_min*60*1500, end_min*60*1500
    tmp = np.zeros((60*1500*(end_min-start_min) - 1), dtype=np.complex128)
    rel_v = get_rel_vests(tmp, v, chunk_len, start_min, end_min)
    tmp = 0 # clear the array

    """
    Convert to a range estimate
    """
    time = np.linspace(0, (start_min-end_min)*60 - 1/1500, rel_v.size)
    r0 = 7500 #it doesn't need to be accurate for the simulatio
    dt = 1/1500
    r = r0 + dt*np.cumsum(rel_v)
    """ now compute doppler shifted frequency for each mode """
    krs = np.load('npy_files/127_krs.npy')
    shape = np.load('npy_files/127_shapes.npy')
    modal_sum = np.zeros((63, rel_v.size), dtype=np.complex128)
    omeg_s = 2*np.pi*freq
    for j in range(63):
        print('simming on sensor ', j)
        for i,kr in enumerate(krs):
            amp  =shape[0,i]*shape[j,i]
            modal_sum[j,:] += amp*np.exp(-complex(0,1)*kr*r)
    #modal_sum += np.exp(complex(0,1)*2*np.pi*127*time)
    modal_sum /= np.sqrt(r)
    modal_sum *= np.exp(complex(0,1)*omeg_s*time)
    np.save('npy_files/127_doppler_sim.npy', modal_sum)

def local_analysis():
    """ just a little sim """
    
    freq = 127
    chunk_len = 1024

    """ get the ship estimates """
    i, t, r, v =load_ship_ests(9, chunk_len,ship_root='../ship/')
    #x = np.load('npy_files/127_ts.npy')
    start_min, end_min = 5, 30
    ind1, ind2 = start_min*60*1500, end_min*60*1500
    #x = x[ind1:ind2]
    #x= hilbert(x)
    #print(x.shape)
    x = np.zeros(60*1500*(end_min-start_min) - 1, dtype=np.complex128)
    rel_v = get_rel_vests(x, v, chunk_len, start_min, end_min)
    print(np.var(rel_v))
    x = 0
    f = freq*(1-rel_v/1456)


    """ now compute doppler shifted frequency for each mode """
    krs = np.load('npy_files/127_krs.npy')
    shape = np.load('npy_files/127_shapes.npy')
    modal_sum = np.zeros(rel_v.size, dtype=np.complex128)
    print(rel_v.size/60/1500)
    time = np.linspace(0, (start_min-end_min)*60 - 1/1500, modal_sum.size)
    for i,kr in enumerate(krs):
        #f = freq*(1-rel_v/1500)
        amp  =shape[0,i]*shape[1,i]
        omeg_r = 2*np.pi*freq - rel_v * kr
        modal_sum += amp*np.exp(-complex(0,1)*omeg_r*time)
    #modal_sum += np.exp(complex(0,1)*2*np.pi*127*time)
    
    mean_f = np.mean(f)
    f_rs = freq - np.mean(rel_v)*krs.real/2/np.pi
    print('mean f', mean_f)
    dev_f = f- mean_f
    freqs, fftvals = np.fft.fftfreq(modal_sum.size, 1/1500), np.fft.fft(modal_sum)
    t = int((end_min-start_min)*60)
    ind1 = int(127.1*t)
    ind2 = int(127.3*t)
    freqs = freqs[ind1:ind2]
    fftvals = fftvals[ind1:ind2]
    psd_orig = np.square(abs(fftvals))
    plt.figure()
    plt.plot(freqs, psd_orig)
    peak_inds = [np.argmin(abs(x-freqs)) for x in f_rs]
    plt.scatter(f_rs, psd_orig[peak_inds], color='r')
    plt.show()

    mean_v = np.mean(rel_v)
    dev_v = rel_v - mean_v
    i = 0
    plt.plot(dev_f)
    plt.show()
    for kr in krs.real:
        gamma = kr / np.mean(krs.real)
        print(np.mean(gamma))
        fr = f_rs[i]
        i += 1
        rel_phase = np.exp(complex(0,-1)*2*np.pi*gamma*dev_f*time)
        print('fr', fr)
        data = rel_phase*modal_sum
        freqs, fftvals = np.fft.fftfreq(data.size, 1/1500), np.fft.fft(data)
        freqs = freqs[ind1:ind2]
        fftvals = fftvals[ind1:ind2]
        psd = np.square(abs(fftvals))
        ind = np.argmin(abs(fr - freqs))
        plt.plot(freqs, psd)
        peak_inds = [np.argmin(abs(x-freqs)) for x in f_rs]
        plt.scatter(f_rs, psd_orig[peak_inds], color='r')
        plt.scatter(freqs[ind], psd[ind], color='b')
        plt.plot(freqs, psd_orig)
        plt.show()
    #f = get_f(gamma, theta)
    #data1 = f*modal_sum
    #np.save('npy_files/127_1.0_20_25.npy', data)

def get_relevant_fhat(fhat, err, start_min, end_min, delta_n):
    """ Extract the best estimates of fhat
        Input 
        fhat - np array
            m by m array of frequency estimates
        err  - np array 
            m by n array of squared error for that corr. 
            f hat
        start_min - float
            between 0 and 74
        end_min - float
            between 0 and 74
        delta_n - int
            number of samples between
            adjacent fhat estimates
            Should eventually package this into an
            object with fhat and err but this is simple enough

        Output 
        fhat - fhat, restricted to start_min to end_min
            and 1 dimensional
    """
    best_inds = np.argmin(err, axis=0)
    i = np.linspace(0, best_inds.size-1, best_inds.size, dtype=int)
    fhat = fhat[best_inds,i]
    """ Restrict to domain of interest """
    ind1, ind2 = int(start_min*60*1500/delta_n), int(end_min*60*1500/delta_n)+1
    fhat = fhat[ind1:ind2]
    return fhat

def interp_fhat(fhat, start_min, end_min, delta_n):
    """
    Interpolate fhat onto the fs=1500 time grid 
    fhat is on a 0.5s grid by default, and generally on
    a grid that is delta_n/1500 seconds apart
    Output
    f_vals - interpolated fhat onto the fs=1500 grid
    """
    fhat_tgrid = np.arange(start_min*60, end_min*60 + delta_n/1500, delta_n/1500)
    f_func = interp1d(fhat_tgrid, fhat)
    time_domain = np.arange(start_min*60, end_min*60, 1/1500)
    f_vals = f_func(time_domain)
    return f_vals

def rescale_f_vals(f_vals_curr, f_vals):
    ratio = np.mean(f_vals_curr)/np.mean(f_vals)
    f_vals *= ratio
    return f_vals

def get_relevant_data(freq, start_min, end_min):
    print(get_fname(freq))
    x = np.load(get_fname(freq))
    ind1, ind2 = int(start_min*60*1500), int(end_min*60*1500)
    x = x[:,ind1:ind2]
    print('len of ts (mins)', x.shape[1] / 1500/60)
    print('start and end min', start_min, end_min)
    return x 

def get_f_grid(freq, mean_f, df):
    """
    Input
    freq - int
        source frequency (used ot compute expected bandwidth
    mean_f - float
        mean doppler shifted received signal
    df - float
        the fundamental frequency
    """
        
    """ Setup frequency search grid"""
    B = get_bw(freq) # bandwidth to search ove/gamm
    """ Why B/4, not B/2? you ask?  My B calculation was very conservative, so B/2 tends to give 
    a much wider range of frequencies than actually have signal. B/4 is still plenty broad """
    f_min = mean_f - B/4
    f_max = mean_f + B/4
    f_grid = np.arange(f_min, f_max, df) 
    return f_grid

def window_x(x):
    """
    Apply a hanning window to x
    """
    N = x.shape[1]
    window = np.hamming(N)
    x = x*window
    return x

def get_mode_mat(freq):
    x = np.load('/home/fakins/code/swellex/audio/npy_files/'+str(freq) + '_mode_mat.npy')
    return x

def est_range_env(x, t):
    """
    Given data x, estimate the range envelope
    """
    sensor0 = hilbert(x[0,:])
    num_samps = sensor0.size
    chunk_size = 40*1500
    chunk_spacing = 20*1500
    num_chunks = (num_samps - chunk_size) // chunk_spacing
    print('num chunks', num_chunks)
    chunk_inds = [slice(chunk_spacing*i, chunk_spacing*i+chunk_size) for i in range(num_chunks)]
    amps = []
    times = []
    for chunk in chunk_inds:
        vals = abs(sensor0[chunk])
        mid_time = np.median(t[chunk])
        amp_est = np.max(vals)
        amps.append(amp_est)
        times.append(mid_time)
    plt.figure()
    plt.plot(times,amps)
    plt.savefig('amp.png')
    
    return

def make_data_fig(freq, t, x):
    fig = plt.figure()
    plt.plot(t/60, x[0,:])
    plt.savefig(str(freq) + '_dats.png')
    plt.close(fig)
    return

def make_fvals_plot(freq, f_vals, f_vals_curr):
    fig =plt.figure()
    plt.plot(f_vals)
    plt.plot(f_vals_curr)
    plt.savefig(str(freq) + '_comp.png')
    plt.close(fig)
    return

def restrict_vals(t, x, f_vals, f_vals_curr):
    bad_inds = np.array([i for i in range(len(t)) if not good_time(t[i])])
    t[bad_inds] = 0 
    x[:,bad_inds] = 0
    f_vals[bad_inds] = 0
    f_vals_curr[good_inds] = 0
    return t, x, f_vals, f_vals_curr

def save_figure(domain, vals, title, save_string):
    """
    Save a simple plot of vals
    """
    fig =plt.figure()
    if type(domain) != type(None):
        plt.plot(domain, vals)
    else:
        plt.plot(vals)
    plt.suptitle(title)
    plt.savefig(save_string)
    plt.close(fig)
    return

def fest_demod(freq, gammas, ref_freq=385, lim_list=[[15,40]], N=1500, delta_n=750, suffix='nb', folder_root='/oasis/tscc/scratch/fakins/', deep=False,mode_filter=False, svd=False, window=False):
    """
    Using ESPRIT freq estimation with narrower filters
    to try and look at spectrum
    freq - int
        source freq
    gammas - list of floats
        gamma vals to search over
    ref_freq - int
        source freq to use f ests for to remove doppler
    lim_list - list of lists of floats
        each list element gives the start and end min of data chunk to look at
        (default looks at one chunk from 15 min to 40 min)
    N - int
        length of data used in the instant. frequ. estimation
    delta_n int
        spacing of chunks used in inst. freq. estimation
    suffix - string
        something to append to the output files for identification
    folder_root - string
        place to save output files
    """
    print('Reference freq', ref_freq)

    for chunk_id, lim in enumerate(lim_list):
        chunk_id = str(chunk_id)
        print(chunk_id)
        start_min = lim[0]
        end_min = lim[1]

        """ Load, select, and interpolate the f ests for freq """
        fhat, err, amp = load_fest(freq,N=N, delta_n=delta_n, tscc=True)
        fhat = get_relevant_fhat(fhat, err, start_min, end_min, delta_n)
        f_vals_curr = interp_fhat(fhat, start_min, end_min, delta_n)

        """ Repeat for the reference freq """
        ref_fhat, ref_err, ref_amp = load_fest(ref_freq,N=N, delta_n=delta_n, tscc=True)
        ref_fhat = get_relevant_fhat(ref_fhat, ref_err, start_min, end_min, delta_n)
        f_vals = interp_fhat(ref_fhat, start_min, end_min, delta_n)

        """ Load data and restrict to start_min, end_min """
        t = np.arange(start_min*60, end_min*60, 1/1500)
        x = get_relevant_data(freq, start_min, end_min)

        """ If SVD flag is set, do a PCA filter """
        if svd==True:
            U, s, Vh = la.svd(x, full_matrices=False)
            print(U.shape, s.shape, Vh.shape)
            count = 0
            while s[count] > .1:
                count += 1
            print(count)
            U = U[:,:count]
            s = s[:count]
            Vh = Vh[:count, :]
            x = U@np.diag(s)@Vh
            print(x.shape)
            save_figure(None, s, 'SVD', str(freq) + '_sing_vals.png')


        """ If window flag is set, window the data """
        if window==True:
            x = window_x(x)


        """ Zero out bad section of data """
        if deep == True:
            t, x, f_vals, f_vals_curr = restrict_vals(t, x, f_vals, f_vals_curr)

        """ Save a visual of the data for checks"""
        save_figure(t, x[0,:], 'Data on sensor1', str(freq) + '_dats.png')

        """ Now scale the reference freq to the freq of interest """
        f_vals = rescale_f_vals(f_vals_curr, f_vals)


        """ Save visual of f_vals """
        make_fvals_plot(freq, f_vals, f_vals_curr)


        """ If mode filtering """
        if mode_filter==True:
            mode_mat = get_mode_mat(freq)
            x = np.linalg.pinv(mode_mat)@x
            print(x.shape, mode_mat.shape)
        
        """ Now estimate the phase noise due to accelerations """
        mean_f = np.mean(f_vals)
        dev_f = f_vals - mean_f 
        save_figure(None, dev_f, 'dev f', str(freq) + '_dev_f.png')
        Theta = get_Theta(dev_f)
        save_figure(None, Theta, 'Theta', str(freq) + '_Theta.png')

        """ Setup frequency search grid"""
        T = (end_min -start_min) * 60
        df = 1/T/5
        f_grid = get_f_grid(freq, mean_f,df)

        np.save(folder_root  + 'f_grid' + '.npy', f_grid)

        """ Gamma iter will remove the phase noise and estimate the PSD at the values supplied in f_grid"""
        fmat = get_f_mat(f_grid, start_min, end_min,deep=deep) 
            
        for gamma in gammas:
            F = get_F(gamma, Theta) # this is the phase noise removal factor
            data = x*F
            vals = fmat@data.T
            vals = vals.T
            peaks = np.zeros((63, 1)) # dummy ting if i ever implement ESPRIT
            output = DemodDat(freq, ref_freq, mean_f, df, gamma, vals, start_min, end_min, N, delta_n, peaks)
            output.save(chunk_id, suffix, folder_root=folder_root)
    return

def fest_baseband(freq, ref_freq=385, lim_list=[[15,40]], N=1500, delta_n=750, folder_root='/oasis/tscc/scratch/fakins/', deep=False, naive=False):
    """
    Using ESPRIT freq estimation with narrower filters
    to correct for the Doppler broadening. 
    Then estimate and compensate for range spreading. 
    Then complex baseband the signal
    freq - int
        source freq
    gammas - list of floats
        gamma vals to search over
    ref_freq - int
        source freq to use f ests for to remove doppler
    lim_list - list of lists of floats
        each list element gives the start and end min of data chunk to look at
        (default looks at one chunk from 15 min to 40 min)
    N - int
        length of data used in the instant. frequ. estimation
    delta_n int
        spacing of chunks used in inst. freq. estimation
    folder_root - string
        place to save output files
    deep - Bool
        flag to denote whether the signal is from deep (54 m) source
        by default it's assumed to be from the shallow source
    naive - Bool
        flag to denote whether or not to use phase corrections
    """
    print('Reference freq', ref_freq)
    for chunk_id, lim in enumerate(lim_list):
        chunk_id = str(chunk_id)
        print(chunk_id)
        start_min = lim[0]
        end_min = lim[1]

        """ Load, select, and interpolate the f ests for freq """
        fhat, err, amp = load_fest(freq,N=N, delta_n=delta_n, tscc=True)
        fhat = get_relevant_fhat(fhat, err, start_min, end_min, delta_n)
        f_vals_curr = interp_fhat(fhat, start_min, end_min, delta_n)

        """ Repeat for the reference freq """
        ref_fhat, ref_err, ref_amp = load_fest(ref_freq,N=N, delta_n=delta_n, tscc=True)
        ref_fhat = get_relevant_fhat(ref_fhat, ref_err, start_min, end_min, delta_n)
        f_vals = interp_fhat(ref_fhat, start_min, end_min, delta_n)

        """ Load data and restrict to start_min, end_min """
        t = np.arange(start_min*60, end_min*60, 1/1500)
        x = get_relevant_data(freq, start_min, end_min)


        """ Zero out bad section of data """
        if deep == True:
            t, x, f_vals, f_vals_curr = restrict_vals(t, x, f_vals, f_vals_curr)

        """ Save a visual of the data for checks"""
        save_figure(t, x[0,:], 'Data on sensor1', str(freq) + '_dats.png')

        """ Now scale the reference freq to the freq of interest """
        f_vals = rescale_f_vals(f_vals_curr, f_vals)

        """ Save visual of f_vals """
        make_fvals_plot(freq, f_vals, f_vals_curr)

        """ Now estimate the phase noise due to accelerations """
        mean_f = np.mean(f_vals)
        dev_f = f_vals - mean_f 
        save_figure(None, dev_f, 'dev f', str(freq) + '_dev_f.png')
        Theta = get_Theta(dev_f)
        save_figure(None, Theta, 'Theta', str(freq) + '_Theta.png')


        """ Remove doppler broadening """
        if naive==False:
            F = get_F(1, Theta) # this is the phase noise removal factor
            data = x*F
        else:
            data = x
        
        """ Demodulate """
        lo = np.exp(complex(0,-1)*2*np.pi*mean_f*t)
        vals = lo*data
        print(vals.shape)
    
        """ Low pass filter/Decimate """
        dec_factor = 150
        baseband = decimate(vals, dec_factor, n = 128, ftype='fir')

        dec_t = t[::150]
        print(dec_t.size, baseband.shape)
        #tmp = vals[0,:]
        #tmp1 = baseband[0,:]
        #print(vals.shape, baseband.shape)
        #freqs = np.fft.fftfreq(tmp.size, 1/1500)
        #freqs1 = np.fft.fftfreq(tmp1.size, 1/1500*dec_factor)
        #fig,axes = plt.subplots(2,1)
        #axes[0].plot(freqs, abs(np.fft.fft(tmp)))
        #axes[1].plot(freqs1, abs(np.fft.fft(tmp1)))
        #plt.savefig(str(freq) + '_lp_check.png')
        x = Baseband(baseband, freq, ref_freq, mean_f, start_min, end_min, 1500/dec_factor, dec_t)
        x.save(naive=naive)
    return

def make_fname(folder_root, gamma, chunk_id, suffix=''):
    fname = folder_root + str(gamma)[:6] + '_' + chunk_id + suffix +'.pickle'
    return fname
    
class DemodDat:
    """ Class to hold the results of demodulating the raw ts"""
    def __init__(self, freq, ref_freq, mean_f, df, gamma, vals, start_min, end_min, N, delta_n,esprit_peaks):
        self.freq = freq # freq band under consideration
        self.ref_freq = ref_freq # freq band used to estimate the phase noise
        self.gamma = gamma # scaling param 
        self.mean_f = mean_f # mean doppler shifted freq over range of interest
        self.df = df
        self.vals = vals # complex spectrum values
        self.start_min = start_min # minute marker of data chunk under consideration
        self.end_min = end_min # end minute marker
        self.N = N # length of data used in the frequency estimation
        self.delta_n = delta_n # spacing of data used in freq. est.
        self.esprit_peaks = esprit_peaks

    def get_fgrid(self):
        """ Construct the fgrid  for vals 
        It's nice to store this procedurally snce i'm copying back and forth
        between comps and it's redundant
        """
        fgrid = get_f_grid(self.freq, self.mean_f,self.df)
        return fgrid
    
    def save(self, chunk_id, suffix, folder_root='/oasis/tscc/scratch/fakins/', overwrite=True):
        fname = make_fname(folder_root, self.gamma, chunk_id, suffix)
        if overwrite == False:
            if os.path.isfile(fname):
                print('Already been computed for gamma', gamma, '. Overwrite is set to False, so returning without computing.')
        else:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)

class Baseband:
    """  Hold results of decimating one of the band time series"""
    def __init__(self, arr_vals, freq, ref_freq, mean_f, start_min, end_min, fs, t):
        self.x = arr_vals # nd array of downsampled vals
        self.freq = freq # signal source frequency
        self.ref_freq = ref_freq # freq band used to estimate the phase noise
        self.mean_f = mean_f # mean doppler shifted freq over range of interest
        self.start_min = start_min # minute marker of decimated data
        self.end_min = end_min # end minute marker
        self.fs = fs  # new sampling rate of downsampled data
        self.dt=  1/ fs
        self.t = t # the resampled time values associated with each point
   
    def save(self,folder_root='/oasis/tscc/scratch/fakins/', overwrite=True, naive=False):
        if naive == False:
            fname = folder_root + str(self.freq) + '_baseband.pickle'
        else:
            fname = folder_root + str(self.freq) + '_naive_baseband.pickle'
        if overwrite == False:
            if os.path.isfile(fname):
                print('Already been computed for . Overwrite is set to False, so returning without computing.')
        else:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)



if __name__ == '__main__':
    fest_baseband(49, ref_freq=388, lim_list=[[5, 30]], deep=True)
    print('npothin to do')
