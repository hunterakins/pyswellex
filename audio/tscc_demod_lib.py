import numpy as np
from matplotlib import pyplot as plt
from swellex.ship.ship import get_good_ests, good_time
from scipy.signal import detrend, firwin, convolve, lombscargle, find_peaks, hilbert
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
import sys
import json

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
        deviation from mean frequency f - mean_f 
    theta0 - float (int is fine too) 
        initial phase offset to start at 
        doesn't actually matter if you're doing incoh"""
    Theta =[theta0]
    for i in range(df.size-1):
        Theta.append(Theta[-1] + 2*np.pi*df[i]*1/1500)
    return np.array(Theta)

def get_F(gamma, Theta):    
    """
    For selected value of gamma (ratio of k_m / bar{k} )
    get phase drift removal factor from Theta
    Input 
    gamma - float
    Theta - np array
    """
    return np.exp(complex(0,1)*gamma*Theta)

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
    vals = mat@x.T
    return np.array(vals)

def gamma_iter(gamma, data, Theta, f_grid,chunk_len, freq,prefix='',start_min=0, end_min=40):
    fname = '/oasis/tscc/scratch/fakins/' + str(freq) + '/' + str(gamma)[:6] + '_' + prefix + str(chunk_len) + '.npy'
    if not os.path.isfile(fname):
        print('Computing FTs for gamma = ', gamma)
        #x = np.load('/oasis/tscc/scratch/fakins/' + str(freq) + '_ts.npy')
        #x = x[:,ind1:ind2]
        F = get_F(gamma, Theta)
        x = data*F
        vals = f_search(f_grid, x, start_min, end_min)
        np.save(fname, vals)
    else:
        print('Already been computed for gamma', gamma, '. Returning')
    return


def save_demod(freq, gamma, deep=False,chunk_len=1024, start_min=0, end_min=40,prefix=''):
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
    np.save('/oasis/tscc/scratch/fakins/' + str(freq) + '/' + prefix + 'f_grid' + '.npy', f_grid)
    gamma_iter(gamma, x, Theta,f_grid, chunk_len, freq, prefix=prefix, start_min=start_min, end_min=end_min)
    return
        
def mode_filter(freq, deep=False,chunk_len=1024, start_min=0, end_min=40,prefix=''):
    """ Fetch data """
    if deep==True:
        v =np.load('/oasis/tscc/scratch/fakins/' + str(chunk_len) + 'dw_rr.npy')
        #v = np.load('../ship/' + str(chunk_len) + 'dw_rr.npy')
        v = v[:-4]
    else:
        i, t, r, v =load_ship_ests(9, chunk_len)
    x = np.load('/oasis/tscc/scratch/fakins/' + str(freq) + '_ts.npy')
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
    np.save('/oasis/tscc/scratch/fakins/' + str(freq) + '/' + prefix + 'f_grid' + '.npy', f_grid)
    gamma_iter(0, x, Theta,f_grid, chunk_len, freq, prefix=prefix, start_min=start_min, end_min=end_min)
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

if __name__ == '__main__':

    #doppler_sim()
    x = np.load('npy_files/127_doppler_sim.npy')
    print(x.shape)
    x = x[0,:150000]
    plt.plot(x)
    plt.show()

    local_analysis()
    sys.exit(0)
    deep_freqs = [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388]
    #test_script()


    #x = np.load('npy_files/49_mini.npy')
    #save_demod(49, 1, deep=True)

    config_file = sys.argv[1]

    with open('confs/' + config_file, 'r') as f:
        lines = f.readlines()
        json_str = lines[0]
        diction = json.loads(json_str)
    freq = diction['freq']
    gamma = diction['gamma']
    chunk_len = diction['chunk_len']
    print(freq, gamma, chunk_len)
    deep = False
    if freq in deep_freqs:
        deep = True
    track_chunks = [[0, 15], [10, 25], [20,35],[30, 45], [40,55]]
    for i, track_chunk in enumerate(track_chunks):
        save_demod(freq, gamma, deep=deep, chunk_len=chunk_len,start_min=track_chunk[0], end_min=track_chunk[1], prefix='chunk_'+str(i))

