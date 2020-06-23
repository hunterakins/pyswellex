import numpy as np
from matplotlib import pyplot as plt
import pickle
from env.env.envs import factory
from pyat.pyat.readwrite import read_modes
from scipy.signal import detrend, firwin, convolve, lombscargle, find_peaks
from swellex.ship.ship import load_track
from swellex.audio.lse import AlphaEst

'''
Description:
Look at effect of range rate on spectral estimation

Date: 6/1

Author: Hunter Akins
'''


def load_ship_ests(source_depth, chunk_len, npy_name=None,r0=7703):
    """
    Get ship range rate estimations produced in ship folder
    Interpolate them onto a uniform grid (in time) with max range step  
    constrained by max_dr
    Input 
        source_depth - str
            either shallow or deep
        chunk_len - float
            length of data chunks used in least squares velocity estimation
        npy_name - string
            location of the saved phase derived velocity estimates
            if type is None i use the default location
        r0 - float or int
            range value corresponding to first point in the v_ests
        max_dr - float
            interpolate if range steps are greater than max_dr
            
    Output 
        ranges - 1d npy array
            ship positions in meters estimated from the velocity
    """
    if type(npy_name) == type(None):
        npy_name = '../ship/npys/' + str(chunk_len) + 's_' + source_depth + '_range_rate.npy'
    v_ests = np.load(npy_name)
    ranges=  [r0]
    for i in range(len(v_ests)):
        dr = v_ests[i]*chunk_len
        new_r = ranges[-1] + dr
        ranges.append(new_r)
    return np.array(ranges), v_ests

def sim_swellex(freq, source_depth, r, sensor_ind=0):
    """
    Do a simulation at freq , source_depth, at ranges in 
    r (a 1d numpy array) at rcvr depth zr[sensor_ind]
    """

    """ Get the modes """
    env_builder  =factory.create('swellex')
    env = env_builder()
    zs = source_depth
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
    krs = modes.k
    As = modes.phi[0,:]
    phi = modes.phi[1:,:] # throw out source depth

    field = np.zeros(r.size, dtype=np.complex128)
    for i in range(krs.size):
        term = As[i]*phi[sensor_ind,i]*np.exp(-complex(0,1)*krs[i].real*r)/np.sqrt(krs[i])
        field += term
#    field *= 1/np.sqrt(r)
    return field, modes

def make_kr_grid(sim_kr, dkr=.0001):
    """
    Grid of wavenumbers to feed into lombscargle
    """
    kmin = .95 * np.min(sim_kr.real)
    kmax = 1.10*np.max(sim_kr.real)
    kr_ests = np.arange(kmin, kmax, dkr)
    return kr_ests

def pgram(r, field, kr_grid):
    pgram = lombscargle(r, field, kr_grid)
    return pgram

def load_rel_data(freq, sensor_ind, r, chunk_len, v_ests,v_bias):
    """
    Load up the relevant sensor time series
    Use the velocity estimates to complex baseband it
    """

    """ Start with 6.5 to 40 mins """
    sensor = np.load('npy_files/sensor' + str(sensor_ind) + '.npy')
    """
    Complex baseband it 
    """
    num_chunk = sensor.size // (1500*chunk_len)
    t = np.arange(0, chunk_len, 1/1500)
    vals = np.zeros((num_chunk), dtype=np.complex128)
    for i in range(num_chunk):
        v = v_ests[i] 
        f = freq*(1 - v/1500)        
        lo = np.exp(complex(0, 1)*2*np.pi*f*t)
        rel_samp = sensor[i*chunk_len*1500:(i+1)*chunk_len*1500]
        val = lo@rel_samp
        vals[i] = val
    v_ests_cpa = v_ests[i:]
    vals *= np.sqrt(r[:num_chunk])
    start_time = 6.5
    start_ind = int((start_time-6.5) * 6)
    vals = vals[start_ind:]

    """ 
    Now get cpa section """
    r_cpa = r[num_chunk:]
    sensor = np.load('npy_files/sensor' + str(sensor_ind) +'cpa.npy')
    """ Cut off at min 46"""
    end_ind = 16*60*1500
    end_ind = -1
    sensor = sensor[:end_ind]
    num_chunk = sensor.size // (1500*chunk_len)
    t = np.arange(0, chunk_len, 1/1500)
    vals_cpa = np.zeros((num_chunk), dtype=np.complex128)
    for i in range(num_chunk):
        v = v_ests_cpa[i]
        if v > 0:
            v = v*v_bias
        f = freq*(1 - v/1500)        
        lo = np.exp(complex(0, 1)*2*np.pi*f*t)
        rel_samp = sensor[i*chunk_len*1500:(i+1)*chunk_len*1500]
        val = lo@rel_samp
        vals_cpa[i] = val
    print('wowow',r_cpa.size, vals_cpa.size)
    vals_cpa *= np.sqrt(r_cpa[:vals_cpa.size])
    all_vals = np.zeros(vals.size+vals_cpa.size, dtype=np.complex128)
    all_vals[:vals.size] = vals
    all_vals[vals.size:] = vals_cpa
    plt.figure()
    plt.suptitle('Basebanded data using velocity estimates')
#    plt.plot(r[start_ind:vals.size+start_ind], (vals))
    plt.plot(r_cpa[:vals_cpa.size], (vals_cpa))
    return all_vals

def incoh_pgram(freq, ranges, v_ests, chunk_len, kr_grid,v_biases=[1]):
    for i in range(21):
        for v_bias in v_biases:
            vals = load_rel_data(freq, i+1,ranges, chunk_len, v_ests,v_bias)
        plt.show()
        p = pgram(ranges[:vals.size], vals, kr_grid)
        if i == 0:
            incoh_sum = p
        else:  
            incoh_sum += p
    return p
  
def get_rel_range(freq, chunk_len, source_string): 
    ranges,v_ests = load_ship_ests(source_string, chunk_len)
    min_ind = np.argmin(ranges)
    branch1 = ranges[:min_ind]
    branch1 = branch1[:300]
    return branch1, ranges, v_ests

def plot_sim_results(kr_grid, branch1, field, modes):
    krs = modes.k
    p = pgram(branch1, field, kr_grid)
    plt.subplots(2,1)
    plt.subplot(211)
    plt.plot(kr_grid,p)
    plt.scatter(krs, [np.mean(p)*2]*len(krs), color='r')
    As = modes.phi[0,:]
    Ar = modes.phi[1,:]
    exc = abs(As*Ar)
    plt.subplot(212)
    plt.scatter(np.linspace(0, exc.size-1, exc.size), exc[::-1])
    plt.xlabel('Mode number')
    plt.ylabel('Excitation')
    plt.ylim([0, np.max(exc)*1.1])

def get_kr_ests(freqs, source_depth, v_bias=[1]):
    for freq in freqs:
        chunk_len = 10
        if source_depth == 9:
            source_string = 'shallow'
        else:
            source_string = 'deep'

        print(source_string)

        branch1, ranges, v_ests = get_rel_range(freq, chunk_len, source_string)
        field, modes = sim_swellex(freq, source_depth, branch1)
        krs = modes.k
        kr_grid = make_kr_grid(krs)

        p = incoh_pgram(freq, ranges, v_ests, chunk_len, kr_grid, v_bias)
        peak_threshold = .20*np.max(p)
        peak_inds, peak_heights = find_peaks(p, height=peak_threshold)
        kr_ests = np.array([kr_grid[x] for x in peak_inds])

        np.save('npy_files/kr_ests_' + str(freq), kr_ests)
        plt.figure()
        plt.plot(kr_grid,p)
        plt.scatter(kr_ests, [np.mean(p)*2]*len(kr_ests), color='r')
        plot_sim_results(kr_grid, branch1, field, modes)
        plt.show()

def plot_kr_ests(freqs):
    i = 0
    for f in freqs:
        kr_ests = np.load('npy_files/kr_ests_' + str(f)+'.npy')
        plt.scatter([i]*len(kr_ests), kr_ests)
        i+=1
    plt.show()

def get_mode_shapes(kr_ests, freq):
    """
    For a given frequency,  ts
    """
    return

def comp_naive_demod(freq, chunk_len,sensor_ind):
    """
    Using GPS derived velocity, perform a demodulation
    """
    
    range_km, lat, lon, time = load_track()
    """
    Only focus on minute 6 to minute 40 """
    r = 1e3*range_km[6:41]
    times = np.arange(6, 40+1/6, 1/6)
    vels = (r[1:] - r[:-1]) / 60
    sensor = np.load('npy_files/sensor' + str(sensor_ind) + '.npy')
    num_chunk = sensor.size // (1500*chunk_len)
    t = np.arange(0, chunk_len, 1/1500)
    vals = np.zeros((num_chunk), dtype=np.complex128)
    for i in range(num_chunk):
        rel_min = int((i+3)//6)
        v = vels[rel_min] 
        f = freq*(1 - v/1500)        
        lo = np.exp(complex(0, 1)*2*np.pi*f*t)
        rel_samp = sensor[i*chunk_len*1500:(i+1)*chunk_len*1500]
        val = lo@rel_samp
        val *= np.sqrt(r[rel_min])
        vals[i] = val
    return vals

    
     
freqs = [109, 127, 145, 163, 198, 232, 280, 335, 385]
source_depth = 9
#get_kr_ests(freqs, source_depth)

freqs = [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388]
source_depth = 54
freqs = [127]
freqs = [335]
freqs = [49]
source_depth = 9
biases=[1, 1.1, 1.2,1.5]
for f in freqs:
    vals = comp_naive_demod(f, 10, 1)
    plt.figure()
    plt.plot((vals[::-1]))
    plt.suptitle('GPS to demodulate')
    get_kr_ests([f], source_depth, biases)


plot_kr_ests(freqs)



