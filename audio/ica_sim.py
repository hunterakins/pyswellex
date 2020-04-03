import numpy as np
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy.signal import firwin, convolve, stft, detrend, decimate, get_window
import scipy
from scipy.io import wavfile
from swellex.ship.ship import load_track, get_velocity
from env.env.envs import factory
from pyat.pyat.readwrite import read_shd, read_modes
from myca.lib.ml_sine import MLOptimize
import os
from myca.lib.mode_analysis import compare_modes, match_modes, switch_sign, normalize_modes
from swellex.ship.ship import load_track, get_velocity
import pickle


"""
Description:
Run Independent Component Analysis on a simulation of a portion of the SwellEx 96 Experiment dataset.
The ship track corresponding to minutes 6.5 to 40 of the S9 source tow is used.
Frequencies 49, 64, and 79 work well.
Simulation results are compared to data.

Author: Hunter Akins

Date: January 2020
"""

def simulate_swellex(freq, saved=False):
    """
    Simulate the data from minutes 6.5 to 40 of the SwellEx dataset at the freq
    of interest. Use normal mode solution to helmholtz and tack on time dependence
    Input -
        freq - int or float
        Frequency of interest. Must be in [49, 64,79, 94, 112,115,127,129]

    Output -
        p - numpy ndarry
            Output of normal mode model up to normalization constant
        pos - Pos object from pyat.env
            Grid on which the p's are computed
        modes - Modes object from pyat.env
            Output of model
        field_time - numpy ndarray
            Field at the swellex receivers with a time dependence added
        stft_field - numpy ndarray
            Output of stft for each sensor. Each row is stft output for that sensor
            Use a Hann window and window length of 1024 samples (a bit less than 1 sec)
    """
    if saved ==True: # already saved
        header =  'npy_files/' + str(int(freq)) + '_'
        p = np.load(header + 'p' + '.npy')
        with open(header+'pos.pickle', 'rb') as f:
            pos = pickle.load(f)
        with open(header + 'modes.pickle', 'rb') as f:
            modes = pickle.load(f)
        stft_times = np.load(header + 'stft_times' + '.npy')
        stft_field = np.load(header + 'stft_field' + '.npy')
        range_vals_meters = np.load(header + 'range_vals_meters' + '.npy')
        return p, pos, modes, stft_times, stft_field, range_vals_meters
    env_builder  =factory.create('swellex')
    env = env_builder()
    zs = 54
    zr = np.array([94.125, 99.755, 105.38, 111.00, 116.62, 122.25, 127.88, 139.12, 144.74, 150.38, 155.99, 161.62, 167.26, 172.88, 178.49, 184.12, 189.76, 195.38, 200.99, 206.62, 212.25])
    num_rcvrs = zr.size
    dz =  5
    zmax = 216.5
    dr = 1
    rmax = 10*1e3

    """
    load ship track
    """
    range_km, lat, lon, time = load_track()
    range_km.shape
    time.shape
    range_interp = scipy.interpolate.interp1d(time[:,0]*60, range_km[:,0])
    times = np.linspace(0, 34*60, 1500*34*60)
    range_vals = range_interp(times)
    range_vals = range_vals[::-1]
    range_vals.size

    """
    Compute modal matrix
    """
    folder = 'at_files/'
    fname = 'swell'
    env.add_source_params(freq, zs, zr)
    env.add_field_params(dz, zmax, dr, rmax)
    p, pos = env.run_model('kraken', folder, fname, zr_range_flag=False)
    modes = read_modes(**{'fname':folder+fname+'.mod', 'freq':freq})
    phi = modes.phi[1:,:] # throw out source depth
    krs = modes.k
    phi.shape

    """
    Compute range dependence over range range_vals and compute field
    """
    range_vals_meters = 1000*range_vals
    for i in range(phi.shape[1]): # for each mode
            mode_weight = modes.phi[0,i] # source weighting
            range_dep = mode_weight*np.exp(complex(0,-1)*krs[i]*range_vals_meters)/np.lib.scimath.sqrt(krs[i])
            range_dep = np.matrix(range_dep)
            vert_dep = np.matrix(phi[:,i]).T
            tmp  =vert_dep @ range_dep
            tmp.shape
            if i == 0:
                field = tmp
            else:
                field += tmp

    weight = np.exp(-complex(0,1)*np.pi/4) /np.sqrt(8*np.pi*range_vals_meters)
    weight.shape
    field.shape
    field = np.array(field)*weight
    sensor1 = field[0,:]

    """
    Compare to Kraken
    (It's worth mentioning why I didn't use Kraken to compute the field.
    They have a limited range resolution that doesn't really make sense.
    With the kr and the modal matrix I can more efficientyl compute the field it seems)
    """
    #env1 = env_builder()
    #env1.add_source_params(freq, zs, [zr[0]])
    #env1.add_field_params(dz, zmax, dr, rmax)
    #p, pos = env1.run_model('kraken', folder, fname, zr_range_flag=False, custom_r=range_vals[::1000])
    #plt.subplot(211)
    #plt.plot(abs(p))
    #plt.subplot(212)
    #plt.plot(abs(.1*sensor1[::1000]))

    """
    Add time dependence
    times defines the time of each snapshot
    assume phase coherence
    """
    time_dependence = np.exp(complex(0,1)*2*np.pi*freq*times)
    #plt.plot(time_dependence)
    field_time = field*time_dependence
    real_field = field_time.real
    #plt.figure()
    #plt.plot(times[::100]/60, field[0,::100])
    #plt.plot(times[::100])

    """
    Use STFT to simulate the processing I perform on the real data
    Data is too big for my computer to do it in one go, so do it in loop
    """
    block_size = 1024
    for i in range(21):
        f,t,z = stft(real_field[i,:], fs=1500, window='hamming', nperseg=block_size, nfft=block_size*4, noverlap = block_size//2, return_onesided=True)
        if i == 0:
            stft_field = np.zeros((21, t.size), dtype=np.complex128)
            diffs = [abs(freq - x) for x in f]
            f_ind = np.argmin(diffs)
            f[f_ind]
        stft_field[i,:] = z[f_ind,:]

    stft_times = t

    header =  'npy_files/' + str(int(freq)) + '_'
    np.save(header + 'p' + '.npy', p)
    with open(header+'pos.pickle', 'wb') as f:
        pickle.dump(pos, f)
    with open(header + 'modes.pickle', 'wb') as f:
        pickle.dump(modes, f)
    np.save(header + 'stft_times' + '.npy', stft_times)
    np.save(header + 'stft_field' + '.npy', stft_field)
    np.save(header + 'range_vals_meters' + '.npy', range_vals_meters)
    return p, pos, modes, stft_times, stft_field, range_vals_meters

def kr_plot(t, range_vals_meters, stft_field):
    mean_dr = np.mean(range_vals_meters[1:] - range_vals_meters[:-1])
    stft1 = stft_field[0,:]
    plt.figure()
    plt.plot(stft1)
    plt.suptitle('STFT of first sensor')
    delta_t = t[1]-t[0]
    delta_r = delta_t * 2.4
    delta_r
    lam_max = 2 * delta_r
    2*np.pi*1/ lam_max
    kr_grid = 2*np.pi*np.fft.fftfreq(stft1.size, delta_r)
    k_vals = np.fft.fft(stft1)
    plt.figure()
    plt.plot(kr_grid, abs(k_vals))
    plt.xlabel('kr')
    plt.ylabel('Abs of spectrum')
    plt.suptitle('FFT of stft of first sensor')

def plot_sensor1_fft(p, range_vals_meters):
    mean_dr = np.mean(range_vals_meters[1:] - range_vals_meters[:-1])
    sensor1 = p[0,:]
    sensor1.size
    real_kr = 2*np.pi*np.fft.fftfreq(sensor1.size, mean_dr)
    plt.figure()
    plt.plot(real_kr, abs(np.fft.fft(sensor1)))
    plt.xlabel('kr')
    plt.ylabel('Abs of fft')
    plt.suptitle('FFT of time domain simulation on first sensor')

def glue_rc(stft_field):
    new_length = stft_field.shape[1]*2
    rc_field = np.zeros((21, new_length))
    rc_field[:,:stft_field.shape[1]] = stft_field.real
    rc_field[:,stft_field.shape[1]:] = stft_field.imag
    return rc_field

def compare_sim_data(data, stft_field):
    fig = plt.figure()
    plt.plot(data[0,::1].real)
    plt.plot(stft_field[0,:5890].real)
    plt.xlabel('Index of sample in time (total interval is 35 mins)')
    plt.suptitle('Comparison of data and simulation (STFT)')
    return fig

def compare_kr_sim_data(data, stft_field):
    window = get_window('hamming', 5890)
    plt.figure()
    plt.plot(abs(np.fft.fft(data[0,:])))
    plt.plot(abs(np.fft.fft(window*stft_field[0,:5890])))
    plt.xlabel('Index of freq bin')
    plt.ylabel('Abs value of spectrum')
    plt.suptitle('Comparison of kr domain of simulation and data on sensor 1')

def get_filtered_data(data, kr_filter_vals, freq, saved=False):
    if saved == True:
        filtered_data = np.load('npy_files/' + str(int(freq)) + '_kr_filtered_data.npy')
        return filtered_data
    fdata = np.fft.fft(data, axis=1)
    plt.plot(abs(fdata[0,:]))
    filtered_fdata = fdata
    filtered_fdata[:,:kr_filter_vals[0]] = 0
    filtered_fdata[:,kr_filter_vals[1]:] = 0
    filtered_data = np.fft.ifft(filtered_fdata, axis=1)
    np.save('npy_files/' + str(int(freq)) + '_kr_filtered_data.npy', filtered_data)
    return filtered_data

def plot_filtered_data(filtered_data):
    plt.figure()
    plt.plot(filtered_data[0,::1])
    plt.suptitle('Filtered data back in STFT domain')

def run_ica_sim(sim_data, num_signals, num_eigs, modes, saved=False):
    if saved==True:
        A = np.load('npy_files/' + str(int(freq)) + '_A_sim.npy')
        return A
    rc_data =glue_rc(sim_data)
    ml_data = MLOptimize(rc_data, num_signals=num_signals)
    B_data = ml_data.compute_B(**{'num_eigs':num_eigs, 'tol':1e-4})
    A_d, s_d = ml_data.estimate()
    compare_modes(modes.phi[1:], A_d, **{'method':'corr'})

    np.save('npy_files/' + str(int(freq)) + '_A_sim', A_d)
    return A_d

def run_ica_data(data, filtered_data, num_signals, num_eigs, modes, saved=False):
    if saved==True:
        A_fd = np.load('npy_files/' + str(int(freq)) + '_A_fd.npy')
        A = np.load('npy_files/' + str(int(freq)) + '_A.npy')
        return A, A_fd
    rc_data =glue_rc(data)
    ml_data = MLOptimize(rc_data, num_signals=num_signals)
    B_data = ml_data.compute_B(**{'num_eigs':num_eigs, 'tol':1e-4})
    A_d, s_d = ml_data.estimate()
    compare_modes(modes.phi[1:], A_d, **{'method':'corr'})

    rc_filtered_data = glue_rc(filtered_data)
    ml_filtered_data = MLOptimize(rc_filtered_data, num_signals=num_signals)
    B_filtered_data = ml_filtered_data.compute_B(**{'num_eigs':num_eigs, 'tol':1e-5})
    A_fd, s_fd = ml_filtered_data.estimate()

    np.save('npy_files/' + str(int(freq)) + '_A_fd', A_fd)
    np.save('npy_files/' + str(int(freq)) + '_A', A_d)
    return A_d, A_fd

def remove_bad_sections(data):
    """
    I observed that the bands 49, 64, 79 and 94 all exhibit
    a small amplitude portion in the following indices
    Perhaps removing these will improve the ICA estimates
    """
    bad_section1 = 1850, 2100
    bad_section2 = 2500, 2775
    bad_section3 = 5520, 5770
    slice_list = [x for x in range(data.shape[1]) if (x < bad_section1[0]) or (x>bad_section1[1] and x<bad_section2[0]) or (x > bad_section2[1] and x < bad_section3[0]) or (x > bad_section3[1])]
    plt.figure()
    plt.plot(slice_list)
    return data[:,slice_list]
    
    

if __name__ == '__main__':
    """
    Configure simulation environment
    """
    freq=94
    freq = freq *(1+2.5/1500)
    kr_filter_list = [[4400, 4460], [5140, 5240], [0,125], [760,860]]
    freq_list = [49, 64, 79, 94]
    freq_list = [x*(1+2.5/1500) for x in freq_list]
    filter_ind = [i for i in range(len(freq_list)) if freq_list[i] == freq][0]
    kr_filter_vals = kr_filter_list[filter_ind]
    p, pos, modes, stft_times, stft_field, range_vals_meters = simulate_swellex(freq, saved=True)
    t = stft_times # more convenient variable name

    kr_plot(t, range_vals_meters, stft_field)
    plt.plot(t/60, stft_field[0,:])
    plot_sensor1_fft(p, range_vals_meters)
    """ Simulation ICA """
    #rc_field = glue_rc(stft_field) # add complex part to end of real
    #ml = MLOptimize(rc_field, num_signals=num_signals)
    #B = ml.compute_B(**{'num_eigs':num_eigs, 'tol':1e-4})
    #A, s = ml.estimate()
    for i in [13]:
        num_signals =  i
        num_eigs  = i
        A = run_ica_sim(stft_field, num_signals, num_eigs, modes, saved=True)
        print('num modeled modes', modes.phi.shape[1])
        compare_modes(modes.phi[1:], A, **{'method':'corr'})
        plt.savefig('pics/' + str(int(freq)) + '_' +  str(i) + 'sim_mode_comp1.png')
        compare_modes(A, modes.phi[1:],**{'method':'corr'})
        plt.savefig('pics/' + str(int(freq)) + '_' + str(i) + 'sim_mode_comp2.png')

        """ Load in data to compare """
        data = np.load('npy_files/' + str(int(freq)) +'_ts_filtered.npy')

        """ Compare stft fields of sim and data """
        compare_sim_data(data, stft_field)

        """ Compare kr domain of each """
        compare_kr_sim_data(data, stft_field)

        """ Filter data in the kr domain """
        filtered_data = get_filtered_data(data, kr_filter_vals, freq, saved=True)

        plot_filtered_data(filtered_data)
        filtered_data = remove_bad_sections(filtered_data)
        compare_sim_data(filtered_data, stft_field)
        compare_sim_data(stft_field, filtered_data)
        plt.savefig('pics/' + str(int(freq)) + '_filtereddata.png')
        """
        Use 49 Hz result to seed the 64 Hz
        """
        A, A_fd = run_ica_data(data, filtered_data, num_signals, num_eigs, modes, saved=False)
        compare_modes(A_fd, modes.phi[1:,:], method='corr')
        plt.savefig('pics/' + str(int(freq)) + '_' +  str(i) + 'mode_comp1.png')
        compare_modes(modes.phi[1:,:], A_fd, method='corr')
        plt.savefig('pics/' + str(int(freq)) + '_' + str(i) + 'mode_comp2.png')
        compare_modes(A, modes.phi[1:,:], method='corr')
        plt.savefig('pics/' + str(int(freq)) + '_unfiltered.png')
plt.show()

"""
Do depth discrimination
mode_strengths = abs(modes.phi[0,:])
mode_shapes = modes.phi[1:,:] # ignore source depth value
inds = corr_match(A, mode_shapes)
print(inds)
inds[0] = 1 # manual correction
Aest = A[:,inds]
Aest = normalize_modes(Aest)
mode_shapes = normalize_modes(mode_shapes)
mode_strengths
plt.figure()
plt.plot(mode_strengths)
plt.figure()
for i in range(6):
    plt.subplot(321+i)
    plt.plot(Aest[:,i])
    plt.plot(mode_shapes[:,i])

p.shape
A.shape

proj = Aest.T@p
proj.shape
pows= np.var(proj, axis=1)
pows.shape
plt.plot(pows/600)
plt.plot(mode_strengths)

"""
