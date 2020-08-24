import numpy as np
import sys
from matplotlib import pyplot as plt
from swellex.audio.tscc_demod_lib import make_fname, DemodDat, Baseband
from swellex.audio.fest_demod import parse_conf_dict
from swellex.audio.autoregressive import esprit
from swellex.audio.modal_inversion import ModeEstimates
import json
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
    return np.sqrt(np.sum(np.square(abs(est)), axis=0))

def get_frs(freq, f_grid):
    krs = np.load('npy_files/' + str(freq) + '_krs.npy')
    krs = krs.real
    v_est = -(freq - np.mean(f_grid)) / freq * 1500
    frs = (krs-np.mean(krs)) * v_est / 2 / np.pi + np.mean(f_grid)
    return frs

def get_shallow_frs(freq, f_grid):
    krs = np.load('npy_files/' + str(freq) + '_krs_180.npy')
    krs = krs.real
    v_est = -(freq - np.mean(f_grid)) / freq * 1500
    frs = (krs-np.mean(krs)) * v_est / 2 / np.pi + np.mean(f_grid)
    return frs

def plot_db_spec(freq,gamma, chunk_list, chunk_load):
    trunc_lims = slice(100,-100)
    #trunc_lims = slice(0,-1)
    num_chunks = len(chunk_list)
    fig, axes = plt.subplots(num_chunks, 1, sharex=True)
    if num_chunks == 1:
        axes = [axes]
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
        shallow_frs = get_shallow_frs(freq, f)
        frs += -.0002
        shallow_frs += -.0002
        for fr in frs:
            axes[i].plot([fr]*10, np.linspace(-15, 0, 10), '-', alpha=.5, color='k')
            axes[i].set_title(str(x.start_min) + ' to ' + str(x.end_min))
        for fr in shallow_frs:
            axes[i].plot([fr]*10, np.linspace(-15, 0, 10), '-', alpha=.3, color='r')
            axes[i].set_title(str(x.start_min) + ' to ' + str(x.end_min))
        axes[i].set_ylim([-16, .5])
        label = 'With phase '  
    axes[0].legend(['No phase', label],prop={'size':8})
    axes[-1].set_xlabel('Frequency (Hz)')
        #plt.legend(['Using 385 reference', 'Using 109 reference', 'Using no reference', 'Simulated kr, shifted using an assumed sound speed of 1530 m/s'])
    return fig, axes

def plot_esprit(freq, gamma, chunk_list, chunk_load):
    """ """
    num_chunks = len(chunk_list)
    fig, axes = plt.subplots(num_chunks, 1, sharex=True)
    if num_chunks == 1:
        axes = [axes]
    plt.suptitle('Gamma ' + str(gamma)[:6])
    for i in range(num_chunks):
        chunk_id = str(i)
        x = chunk_load(freq, gamma, chunk_id)
        frs = get_frs(freq, x.get_fgrid())
        peaks = x.esprit_peaks
        for j in range(63):
            axes[i].scatter(abs(peaks[j,:]), [j]*peaks.shape[1])
        axes[i].scatter(frs, [64]*len(frs))
    return fig

def chunk_load_dir(freq, gamma,chunk_id, proj_dir, suffix=''):
    """ Load a single demod dat for source frequency freq, gamma val, and chunk_id """
    fname = make_fname('npy_files/' + str(freq) + '/' + proj_dir, gamma, chunk_id, suffix)
    with open(fname, 'rb') as f:
        x = pickle.load(f)
    return x

def animate_figs(freq,proj_dir):
    vid_name = str(freq) + '.mp4'
    os.chdir('pics/' + str(freq) + '/' + proj_dir)
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
        
def algorithm_1(freq, chunk, chunk_load):
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
    peak_inds, _ = find_peaks(pow_db, height=-4)
    peak_heights = power[peak_inds]
    #peak_inds = peak_inds[np.argsort(peak_heights)]
    #peak_heights = np.sort(peak_heights)
    peak_inds = peak_inds
    peak_heights = peak_heights
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
        print(g)
        plt.plot(mode_shape.real, depths)
        plt.plot(mode_shape.imag, depths)
        plt.show()

def algorithm_2(fgrid, fft, power, num_peaks=4, distance=10):
    """
    Just pick the peaks...i was originally searching
    over gamma pick num_peaks biggest peaks
    """
    norm = np.max(power)
    power /= norm
    pow_db = 10*np.log10(power)
    power *= norm
    ref_power = power
    h0 = -4
    peak_inds, _ = find_peaks(pow_db, distance =distance, height=h0)
    while len(peak_inds) < num_peaks:
        h0 -= 1
        peak_inds, _ = find_peaks(pow_db, distance =6, height=h0)
    peak_heights = power[peak_inds]
    peak_inds = peak_inds[np.argsort(peak_heights)[::-1]]
    peak_heights = np.sort(peak_heights)[::-1]
    peak_inds = peak_inds[:num_peaks]
    peak_heights = peak_heights[:num_peaks]
    best_inds = peak_inds
    best_inds.sort()
    modes = fft[:,best_inds]
    return modes, best_inds
            
def track_comparison(freq, gammas, chunk_list, proj_dir):
    """
    For the demodulated data for the recent chunks
    Plot power for each gamma in a big subplot
    """

    #chunk_load(109, gamma, '0')
    if proj_dir[:-1] not in os.listdir('pics/' + str(freq)):
        os.mkdir('pics/' + str(freq) + '/' + proj_dir)
    for i, gamma in enumerate(gammas):
        fig,axes = plot_db_spec(freq, gamma, chunk_list)
        #fig.set_size_inches(32, 18)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        print('pics/' + str(freq) + '/' + proj_dir)
        fig.savefig('pics/' + str(freq) + '/' + proj_dir + str(i).zfill(3), bbox_inches='tight',dpi=500)
        plt.close(fig)

def esprit_comp(freq, gammas, chunk_list, proj_dir):
    """
    """

    #chunk_load(109, gamma, '0')
    if proj_dir[:-1] not in os.listdir('pics/' + str(freq)):
        os.mkdir('pics/' + str(freq) + '/' + proj_dir)
    for i, gamma in enumerate(gammas):
        fig = plot_esprit(freq, gamma, chunk_list)
        #fig.set_size_inches(32, 18)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        print('pics/' + str(freq) + '/' + proj_dir)
        #fig.savefig('pics/' + str(freq) + '/' + proj_dir + str(i).zfill(3), bbox_inches='tight',dpi=500)
        plt.show()
        plt.close(fig)

def mf_filt(freq, gammas, proj_dir, chunk_load):
    gamma = gammas[np.argmin(np.array([abs(x-1) for x in gammas]))]
    x = chunk_load(freq, gamma, str(0))
    f = x.get_fgrid()
    frs = get_frs(freq, x.get_fgrid())
    shallow_frs = get_shallow_frs(freq, x.get_fgrid())
    vals = x.vals
    modes = np.load('npy_files/' + str(freq) + '_mode_mat.npy')
    fig, axes = plt.subplots(2,1)
    shape_list, fis = [], []
    c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    array_inds = np.arange(0, len(modes[:,0]), 1)
    for i in range(vals.shape[0]):
        axes[0].plot(f, abs(vals[i,:]), color=c_list[i])
        axes[0].scatter(frs, [1]*len(frs), alpha=.5, color='k')
        axes[0].scatter(shallow_frs, [1]*len(shallow_frs), alpha=.5, color='r')
        axes[1].plot((modes[:,i]).imag,array_inds, color=c_list[i])
        axes[1].plot((modes[:,i]).real,array_inds, color=c_list[i])
        f_i = f[np.argmax(abs(vals[i,:]))] 
        fis.append(f_i)
        shape_list.append(modes[:,i])
    me = ModeEstimates(freq, shape_list, fis, interp_bad_el=True)
    return me

def run_inv_mod(me):
    """
    Do a smoothing of the shapes, fit to sine model,
    and then invert for vhat and chat """
    me.get_real_shapes()
    me.get_smooth_shapes()
    me.kz_estimate()
    err_threshold = 9
    me.remove_bad_matches()
    v_hat, c_hat = me.run_inv() 
    return v_hat, c_hat

def get_mode_mat(freq, proj_dir, num_modes):
    gammas = np.load(proj_root + proj_dir+'gammas.npy')

    depths = np.linspace(94.125, 212.25, 64)
    depths = np.delete(depths, 21)

    chunk_list = [0]
    chunk_load = lambda freq, gamma, chunk_id: chunk_load_dir(freq, gamma, chunk_id, proj_dir)
    fig, axes = plot_db_spec(freq,1, chunk_list)

    modes, best_inds = algorithm_2(freq, 0, num_modes)

    dat = chunk_load(freq, 1, '0')
    f_grid = dat.get_fgrid()
    axes[0].scatter(f_grid[best_inds], [-3]*len(best_inds), color='r')
    plt.show()
    for i in  range(modes.shape[1]):
        plt.plot(modes[:,i].real)
        plt.plot(modes[:,i].imag)
        plt.show()
    np.save('npy_files/' + str(freq) +'_mode_mat.npy', modes)
    return modes

def copy_results(exp_id):
    conf_file = exp_id + '.conf'
    with open('confs/' + conf_file, 'r') as f:
        lines = f.readlines()
        json_str = lines[0]
        diction = json.loads(json_str)
    freq, exp_id, track_chunks, gammas, mode_filter_flag, svd_flag, window_flag, deep_flag, ref_freq, proj_dir = parse_conf_dict(diction)
    chunk_list = [i for i in range(len(track_chunks))]
    folder_root = 'npy_files/' + str(freq) + '/'
    
    if proj_dir[:-1] in os.listdir(folder_root):
        print('Already copied data')
        return proj_dir, freq, chunk_list
    else:
        remote_root = 'fakins@tscc-login.sdsc.edu:/oasis/tscc/scratch/fakins/'+str(freq) + '/'
        print('Copying data from ' + remote_root)
        os.chdir('npy_files/'+str(freq))
        scp_command = 'scp -r ' + str(remote_root) + proj_dir + ' .'
        print(scp_command)
        os.system(scp_command)
        os.chdir('../../')
        return proj_dir, freq, chunk_list

def compare_tracks(exp_id):
    """ Compare the effect of an SVD filter on the
    PSD estimation """
    
    proj_dir, freq, chunk_list = copy_results(exp_id)
    proj_root = 'npy_files/' + str(freq) + '/'
    
    """ Copy over results """

    """ Iterate through tracks and plot db scale incoh psd sum for each """
    gammas = np.load(proj_root + proj_dir+'gammas.npy')
    depths = np.linspace(94.125, 212.25, 64)
    depths = np.delete(depths, 21)
    chunk_load = lambda freq, gamma, chunk_id: chunk_load_dir(freq, gamma, chunk_id, proj_dir)

    fig1, axes1 = plot_db_spec(freq,1, chunk_list, chunk_load)
    plt.show()


    return 

def load_bb(freq,naive=False):
    bb_name = str(freq) + '_baseband.pickle'
    if naive == True:
        bb_name = str(freq) + '_naive_baseband.pickle'
    if bb_name not in os.listdir('pickles'):
        remote_root = 'fakins@tscc-login.sdsc.edu:/oasis/tscc/scratch/fakins/'
        scp_command = 'scp ' + str(remote_root) + bb_name + ' pickles/'
        os.system(scp_command)
    with open('pickles/' +bb_name, 'rb') as f:
        bb = pickle.load(f)
    return bb

def get_peak_num(freq, scaling=5):
    return int(freq / 49 * scaling) 

   
def get_modal_filter(modes, best_inds): 
    modes /= np.max(abs(modes), axis=0)
    mode_filter = np.linalg.pinv(modes)
    mode_filter = modes.T
    return mode_filter

def make_bb_fgrid(bb):
    mean_f = bb.mean_f
    fft_freq = np.fft.fftfreq(bb.x.shape[1], bb.dt)
    fft_freq += mean_f
    return fft_freq

def make_real(modes):
    num_modes = modes.shape[1]
    for i in range(num_modes):
        angles = np.angle(modes[:,i])
        med_angle = np.median(angles)
        modes[:,i] = (modes[:,i]*np.exp(complex(0,1)*-1*med_angle)).real
    modes = modes.real
    return modes
    
def plot_bb_spec(freq):
    """ 
    A rip off of plot db spec but using the basebanded 
    data"""
    zero_pad_fact  =5
    bb = load_bb(freq)
    print(bb.start_min, bb.end_min, bb.dt)
    est = bb.x
    naive_bb = load_bb(freq, naive=True)
    naive_est = naive_bb.x

    est = np.fft.fft(est, n = est.shape[1]*zero_pad_fact)
    est = np.fft.fftshift(est,axes=-1)
    naive_est = np.fft.fft(naive_est, n=naive_est.shape[1]*zero_pad_fact)
    naive_est = np.fft.fftshift(naive_est, axes=-1)
    est_pow = get_pow(est)
    naive_pow = get_pow(naive_est)
    norm = np.max(est_pow)
    est_db = est_pow / norm
    est_db = 10*np.log10(est_db)
    naive_db = naive_pow / norm
    naive_db = 10*np.log10(naive_db)

    fig, axis = plt.subplots(1,1)

    f_grid = np.fft.fftfreq(est.shape[1], bb.dt)
    f_grid=np.fft.fftshift(f_grid)
    f_grid += bb.mean_f

    df = f_grid[1]-f_grid[0]
    i1 = f_grid.size //2 - int(.005 *freq/49 / df)
    i2 = f_grid.size //2 + int(.005 *freq/49/ df)
    trunc_lims = slice(i1, i2)
    axis.plot(f_grid[trunc_lims], naive_db[trunc_lims], color='b',  alpha=.2)
    axis.plot(f_grid[trunc_lims], est_db[trunc_lims], color='g')

    frs = get_frs(freq, f_grid)
    shallow_frs = get_shallow_frs(freq, f_grid)
    frs += -.0002
    shallow_frs += -.0002
    for fr in frs:
        axis.plot([fr]*10, np.linspace(-15, 0, 10), '-', alpha=1, color='k')
    #for fr in shallow_frs:
    #    axis.plot([fr]*10, np.linspace(-15, 0, 10), '-', alpha=.5, color='r')
    axis.set_ylim([-16, .5])
    axis.legend(['No phase', 'With phase'],prop={'size':8})
    axis.set_xlabel('Frequency (Hz)')
        #plt.legend(['Using 385 reference', 'Using 109 reference', 'Using no reference', 'Simulated kr, shifted using an assumed sound speed of 1530 m/s'])
    num_peaks = get_peak_num(freq)
    modes, best_inds = algorithm_2(f_grid, est, est_pow,num_peaks=num_peaks)
    c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_list = c_list + c_list
    for i, ind in enumerate(best_inds):
        mode = modes[:,i]
        phase = np.median(np.angle(mode))
        mode *= np.exp(complex(0,1) * - 1 * phase)
        mode /= np.max(abs(mode.real))
        mode *= .0005
        axis.plot(mode.real + f_grid[ind], np.linspace(-15, 0, mode.size), color=c_list[i])
        #axis.plot(mode.imag + f_grid[ind], np.linspace(-15, 0, mode.size))
        axis.plot([f_grid[ind]]*mode.size, np.linspace(-15, 0, mode.size), color=c_list[i])
    return modes

def make_mode_plot(f_grid, est_pow, mode_filtered, trunc_lims, c_list, frs, best_inds, modes, num_peaks, zero_pad_fact, mode_inds):
    est_pow /= np.max(est_pow)
    fig, axes = plt.subplots(2,1)
    axes[0].plot(f_grid[trunc_lims], est_pow[trunc_lims], color='k')
    for i in range(num_peaks):
        mode_output = mode_filtered[i,:]
        power = np.fft.fft(mode_output, n=mode_output.size*zero_pad_fact)
        power = np.fft.fftshift(power)
        power = abs(power)
        power /= np.max(power)
        axes[0].plot(f_grid[trunc_lims], power[trunc_lims], color=c_list[i])
        axes[0].scatter(f_grid[best_inds[i]], est_pow[best_inds[i]], color=c_list[i])
        axes[0].scatter(f_grid[mode_inds[i]], power[mode_inds[i]], color='k')
        axes[1].plot(modes[:,i])
    for fr in frs:
        axes[0].plot([fr]*10, np.linspace(0, 1, 10), '-', alpha=1, color='k')

def get_mode_inds(best_inds, mode_filtered, zero_pad_fact):
    """
    Best_inds are the locations of the peaks in the original fft
    Pick the peaks from the mode_filtered data that are closest to 
    the original estimates """
    num_modes = mode_filtered.shape[0]
    new_inds = []
    for i in range(num_modes):
        mode_output = mode_filtered[i,:]
        fval = np.fft.fft(mode_output, n=mode_output.size*zero_pad_fact)
        fval = np.fft.fftshift(fval)
        power = abs(fval)
        peak_inds, _ = find_peaks(power)
        original_ind = best_inds[i]
        diffs = [abs(x - original_ind) for x in peak_inds]
        new_ind = peak_inds[np.argmin(diffs)]
        new_inds.append(new_ind)
    return new_inds

def get_grid(bb, f_est):
    f_grid = np.fft.fftfreq(f_est.shape[1], bb.dt)
    f_grid=np.fft.fftshift(f_grid)
    f_grid += bb.mean_f

    df = f_grid[1]-f_grid[0]
    i1 = f_grid.size //2 - int(.015 *freq/49 / df)
    i2 = f_grid.size //2 + int(.015 *freq/49/ df)
    trunc_lims = slice(i1, i2)
    return f_grid, trunc_lims

def plot_est_psd(f_grid, est_db, frs, trunc_lims):
    fig, axis = plt.subplots(1,1)
    for fr in frs:
        axis.plot([fr]*10, np.linspace(-16, 0, 10), '-', alpha=1, color='k')
    axis.plot(f_grid[trunc_lims], est_db[trunc_lims], color='g')
    axis.set_ylim([-16, .5])
    #axis.legend(['No phase', 'With phase'],prop={'size':8})
    axis.set_xlabel('Frequency (Hz)')
    return fig, axis

def correct_range_spreading(bb):
    """ Correct range spreading of basebanded
    data """
    depth_avg_pow = np.sum(np.square(abs(bb.x)), axis=0)
    depth_avg_pow /= depth_avg_pow[0]
    H = np.ones((bb.t.size, 2))
    H[:, 1] = np.square(bb.t)
    inv = np.linalg.pinv(H)
    p = inv@depth_avg_pow
    X = p[0]
    alpha = p[1]
    forward = H@p
    r = 1/forward
    sqrt_r = np.sqrt(r)
    bb.x = bb.x*sqrt_r
    return

def get_peak_num_pca(bb):
    """
    Use the eigenvalues to estimate numnber of 
    modes """
    est = bb.x
    cov = np.cov(est)
    w, v = np.linalg.eigh(cov)
    w /= np.max(w)
    num_peaks = len([x for x in w if x > .01])
    peak_num_list = [num_peaks]
    print('num_peaks', num_peaks)
    return num_peaks

def get_f_est(bb, zero_pad_fact): 
    """
    Estimate the PSD of the basebanded data
    return f_est as the elementwise fft of the baseband,
    as well as the incoh sum in DB"""
    est = bb.x
    f_est = np.fft.fft(est, n = est.shape[1]*zero_pad_fact)
    f_est = np.fft.fftshift(f_est,axes=-1)
    est_pow = get_pow(f_est)
    #est_pow = np.square(abs(f_est[-1,:]))
    norm = np.max(est_pow)
    est_db = est_pow / norm
    est_db = 10*np.log10(est_db)
    return f_est, est_pow, est_db

def get_modal_peaks(freq):
    """ Load up basebanded data and get incoherent sum
    psd estimate """
    c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_list = c_list + c_list
    zero_pad_fact  =7
    bb = load_bb(freq)
    start_min = 5
    start_ind = int(60*start_min / bb.dt)
    bb.x = bb.x[:,start_ind:]
    bb.t = bb.t[start_ind:]
    print('Time interval', bb.start_min, bb.end_min)

    """ Deal with range spreading """
    correct_range_spreading(bb)

    """ Use eigenvalues of cov mat to estimate number 
    of modal peaks to search for """
    num_peaks = get_peak_num_pca(bb)
    peak_num_list = [num_peaks]
    #peak_num_list = [num_peaks - 1, num_peaks, num_peaks + 1]
  
    """ Estimate PSD by incoh sum """ 
    f_est, est_pow, est_db = get_f_est(bb, zero_pad_fact)

    """ Get the corresponding frequency grid of the PSD."""
    f_grid, trunc_lims = get_grid(bb, f_est)
    df = f_grid[1]-f_grid[0]
    
    """ Fetch simulated doppler shifted modes and an estimate of
    how close they should be """
    frs = get_frs(freq, f_grid)
    min_space = np.min(abs(frs[1:] - frs[:-1]))
    distance = int(min_space/df)
   
    """ Plot incoh sum if desired """ 
    fig, axis = plot_est_psd(f_grid, est_db, frs, trunc_lims)
    plt.show()
    """ Use incoh_sum to get moe estimates """
    #num_peaks = get_peak_num(freq, scaling)
    v_hats, c_hats = [], []
    for num_peaks in peak_num_list:
        modes, best_inds = algorithm_2(f_grid, f_est, est_pow,num_peaks=num_peaks,distance=distance)
        mode_filter = get_modal_filter(modes, best_inds)

        """ Get the f_r from the filtered modes """
        modes = make_real(modes)
        start_min = 0
        start_ind = int(60*start_min / bb.dt)
        bb.x = bb.x[:,start_ind:]
        mode_filtered = mode_filter@bb.x
        for i in range(num_peaks):
            plt.plot(bb.t,mode_filtered[i,:])
        plt.show()
        mode_inds = get_mode_inds(best_inds, mode_filtered, zero_pad_fact)
        make_mode_plot(f_grid, est_pow, mode_filtered, trunc_lims, c_list, frs, best_inds, modes, num_peaks, zero_pad_fact, mode_inds)
        plt.show()

        fr = f_grid[mode_inds]
        shape_list = [modes[:,i] for i in range(num_peaks)]
        """ Throw out first few modes since WKB doesn't work well"""
        #fr = fr[:-1]
        #shape_list = shape_list[:-1]
        me = ModeEstimates(freq, shape_list, fr, interp_bad_el=True)
        v_hat, c_hat = run_inv_mod(me)
        #me.compare_forward_model()
        #me.show_kz_match()
        #plt.plot(me.kzs, me.fi)
        #plt.plot(me.best_kzs, me.best_fi, color='r')
        #plt.show()
        v_hats.append(v_hat)
        c_hats.append(c_hat)
    print(v_hats, c_hats)
    return v_hats, c_hats, peak_num_list
    
def run_multiband_inverse():
    gps_v = -(7703-2196) / (38.5 * 60)
    print('gps v', gps_v)
    ctd_c = 1488.53 # taken from ctd i9605
    print('ctd_c', ctd_c)
    fig, axes = plt.subplots(2,1)
    axes[1].set_xlabel('Number of peaks searched for')
    axes[0].set_ylabel('vhat (m/s)')
    axes[1].set_ylabel('chat (m/s)')
    scale_list = np.arange(5.5, 7.5, .5)
    freqs = [49, 64, 79, 94, 109, 127, 130, 145, 148, 163, 166]#, 166, 198]#, 201, 232]
    #freqs = freqs[::-1]
    c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_list = c_list+ c_list
    min_num = 1e5
    max_num = 0
    full_vhat_list = []
    full_chat_list = []
    for i, freq in enumerate(freqs):
        num_peaks_list = list(set([get_peak_num(freq, x) for x in scale_list]))
        print('freq', freq)
        v_hats, c_hats, _ = get_modal_peaks(freq)
        kbar_hats = [2*np.pi*freq/x for x in c_hats]
        axes[0].scatter([freq]*len(v_hats), v_hats, color=c_list[i])
        axes[1].scatter([freq]*len(v_hats), c_hats, color=c_list[i])
        full_vhat_list.append(np.median(v_hats))
        full_chat_list.append(np.median(c_hats))
    dom = np.arange(min(freqs), max(freqs)+.1, 1)
    axes[1].legend([str(freq) for freq in freqs])
    med_vhat = np.mean(full_vhat_list)
    med_chat = np.mean(full_chat_list)
    axes[0].plot(dom, [med_vhat]*dom.size, color='b')
    axes[0].plot(dom, [gps_v]*dom.size, color='k')
    axes[1].plot(dom, [ctd_c]*dom.size, color='k')
    axes[1].plot(dom, [med_chat]*dom.size, color='b')
    plt.show()

if __name__ == '__main__':
    """
    Parameters are source frequency, project dir, chunk_list 
    source freq - int
    proj_dir - string
        include the trailing / but not the leading /
        assumes data is in npy_files/[freq]/proj_dir/
        and saves pictures to pics/[freq]/proj_dir/
    chunk_list - [0, 1, ..] for each track chiunk you do 
    """
    """ This is a global parameter determined by fest_demod.py"""

    #run_multiband_inverse()
    c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_list = c_list + c_list
    i = 0 
    freqs = [49, 64, 79, 94, 109, 130, 145, 127, 145, 148, 163, 166]
    for freq in freqs:
        get_modal_peaks(freq)
        i += 1
    plt.legend([str(x) for x in freqs])
    plt.show()

