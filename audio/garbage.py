import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import matplotlib._color_data as mcd
import os
from scipy.signal import find_peaks
from swellex.audio.modal_inversion import ModeEstimates, build_model_mat, build_data_mat
from env.env.envs import factory

'''
Description:
Implementing the processing in Shane Walker's paper
to extract the modes from Swellex data

Date: 

Author: Hunter Akins
'''


def get_names(freq, chunk_len=1024,short=False, chunk=None):
    """
    Look through the npy files directory for the saved files
    """
    if short == True:
        names = [x for x in os.listdir(os.getcwd() + '/npy_files/' + str(freq)+ '/'+ 'short') if x[-5] != 'd']
    elif type(chunk) != type(None):
        names = [x for x in os.listdir(os.getcwd() + '/npy_files/' + str(freq)+ '/'+ 'chunk'+str(chunk)) if x[-5] != 'd']
        
    else:
        names = [x for x in os.listdir(os.getcwd() + '/npy_files/' + str(freq)+ '/'+ str(chunk_len)) if x[-5] != 'd']
    return names

def get_gammas_from_names(names,short=False, chunk=None):
    if short == True:
        gammas = [float(x[:-15]) for x in names]
    elif type(chunk) != type(None):
        gammas = [float(x[:-7]) for x in names]
        #gammas = [float(x[:-17]) for x in names]
    else:
        gammas = [float(x[:-7]) for x in names]
    gammas.sort()
    return gammas
    
def get_freqs(freq,chunk=None):
    if type(chunk)==type(None):
        f = np.load('npy_files/' + str(freq) + '/f_grid.npy')
    else:
        #f = np.load('npy_files/' + str(freq) + '/chunk' + str(chunk) + '/chunk_' + str(chunk) + '_f_grid.npy') 
        f = np.load('npy_files/' + str(freq) + '/chunk' + str(chunk) + '/nb_f_grid.npy') 
    return f

def get_dats_for_gamma(gamma, freq, chunk_len=1024,short=False,chunk=None):
    """
    For a value of gamma and frequency freq
    get the search over f provided
    Rturn x
    numpy array
    each row is a frequency
    each column is a sensor
    """
    if short == True:
        tail = '_short_' + str(chunk_len) + '.npy'
    elif type(chunk) != type(None):
        #tail = '_chunk_' + str(chunk) + '_' + str(chunk_len) + '.npy'
        tail = '_nb.npy'
    else:
        tail = '_' + str(chunk_len) + '.npy'
    fname =  str(gamma)[:7]+ tail
    try:
        if short == True:
            x = np.load('npy_files/' +str(freq) + '/short/' + fname)
        elif type(chunk) != type(None):
            x = np.load('npy_files/' +str(freq) + '/chunk' + str(chunk) + '/' + fname)
        else:
            x = np.load('npy_files/' +str(freq) + '/' + str(chunk_len) + '/' + fname)
    except FileNotFoundError:
        try:
            fname = str(gamma)[:7] + '000' + tail
            if short == True:
                x = np.load('npy_files/' +str(freq) + '/short/' + fname)
            elif type(chunk) != type(None):
                x = np.load('npy_files/' +str(freq) + '/chunk' + str(chunk) + '/' + fname)
            else:
                x = np.load('npy_files/' +str(freq) + '/' + str(chunk_len) + '/' + fname)
        except FileNotFoundError:
            try:
                fname = str(gamma)[:7] + '0' + tail
                if short == True:
                    x = np.load('npy_files/' +str(freq) + '/short/' + fname)
                elif type(chunk) != type(None):
                    x = np.load('npy_files/' +str(freq) + '/chunk' + str(chunk) + '/' + fname)
                else:
                    x = np.load('npy_files/' +str(freq) + '/' + str(chunk_len) + '/' + fname)
            except FileNotFoundError:
                fname = str(gamma)[:7] + '00' + tail
                if short == True:
                    x = np.load('npy_files/' +str(freq) + '/short/' + fname)
                elif type(chunk) != type(None):
                    x = np.load('npy_files/' +str(freq) + '/chunk' + str(chunk) + '/' + fname)
                else:
                    x = np.load('npy_files/' +str(freq) + '/' + str(chunk_len) + '/' + fname)
    return x

def get_depths():
    depths = np.linspace(94.125, 212.25, 64)
    depths = np.delete(depths, 21)
    return depths
    
def gen_figs(freq, chunk_len=1024, short=False, chunk=None):
    """ 
    Process all the shit for a given frequency 
    """
    krs = np.load('npy_files/' + str(freq) + '_krs.npy')
    krs = krs.real
    names = get_names(freq,short=short,chunk=chunk)
    gammas = get_gammas_from_names(names,short,chunk=chunk)
    depths=  get_depths()
    f = get_freqs(freq,chunk=chunk)
    mean_f = np.mean(f)
    mean_v = (np.mean(f) - freq)*1450/freq
    print(mean_v)
    kr_ests = mean_f+(krs-np.mean(krs))*mean_v/2/np.pi

    ylim = 0
    for i, gamma in enumerate(gammas):
        fname = str(gamma)[:7]+'.npy'
        x = get_dats_for_gamma(gamma, freq, chunk_len, short=short, chunk=chunk)
        power = np.square(abs(x))
        incoh_sum = np.sum(power, axis=1)
        max_val = np.max(incoh_sum)
        if max_val > ylim:
            ylim = max_val
    ylim *= 1.1

    

    for i, gamma in enumerate(gammas):
        fname = str(gamma)[:7]+'.npy'
        x = get_dats_for_gamma(gamma, freq, chunk_len, short=short, chunk=chunk)
        power = np.square(abs(x))

        fig, (ax1, ax2)=plt.subplots(2,1,sharex='col')
        ax2.set_ylim([0, ylim])
        levels = np.linspace(np.min(power), np.max(power), 20)
        ax1.contour(f, depths[::-1], power.T, levels=levels)
        incoh_sum = np.sum(power, axis=1)
        peak_ind = np.argmax(incoh_sum)
        peak_mode = x[peak_ind,:]
        peak_mode.real /= np.max(abs(peak_mode.real))*.1*np.max(f)
        peak_mode.imag /= np.max(abs(peak_mode.imag))*.1*np.max(f)
        f_offset = f[int(len(f)*.75)]
        peak_mode.real += f_offset
        peak_mode.imag += f_offset
        #ax1.plot(peak_mode.real, depths)
        #ax1.plot(peak_mode.imag, depths)
        ax2.plot(f, np.sum(power, axis=1))
        ax2.scatter(kr_ests, [ylim/10]*kr_ests.size, marker='x',color='k')
        if type(chunk) != type(None):
            fig_name = 'pics/gammapows/' + str(freq) + '/' +str(chunk) +'/'+ str(i).zfill(3) + '.png'
        else:
            fig_name = 'pics/gammapows/' + str(freq) + '/' + str(i).zfill(3) + '.png'
        plt.suptitle('Gamma = ' +str(gamma) +',  Chunk ' + str(chunk))
        plt.xlabel('Frequency (Hz)')
        ax1.set_ylabel('Depth')
        ax2.set_ylabel('Power')
        plt.show()
        plt.savefig(fig_name)
        plt.show()
        plt.close(fig)

def gen_range_dep_movie(freq, chunks, chunk_len=1024, short=False):
    """
    I'm looking into the range dependence of the waveguide
    I perform my analysis (searching over gamma) for each 
    interval in chunks (on the supercomputer)
    Then I want to look at the evolution of the spectrum as the 
    intervals change    
    SO just stack a bunch of subplots
    Input 
    freq - int 
    chunks - list of lists (the chunks in minutes)
    chunk_len - int
    short - bool
    """
    #track_chunks = [[0, 15],[5,20], [10, 25],[15,30], [20,35],[25,40],[30, 45],[35,50],[40,55]]
    rep_chunk = chunks[0]
    names = get_names(freq,short=short,chunk=0)
    gammas = get_gammas_from_names(names,short,chunk=0)
    num_chunks = len(chunks)

    """ Get axis limits """
    smallest ,biggest=1e9,0
    for i in range(num_chunks):
        f_grid = get_freqs(freq,chunk=i)
        small, big = f_grid[0], f_grid[-1]
        if small < smallest:
            smallest = small
        if big > biggest:
            biggest = big

    """ Loop through and create a figure for each gamma """
    for i, gamma in enumerate(gammas):
        fig, axes = plt.subplots(num_chunks, 1)
        for j in range(num_chunks):
            axis = axes[j]
            axis.set_xlim([smallest, biggest])
            try:
                dats = get_dats_for_gamma(gamma, freq, chunk=j)
            except:
                dats = np.zeros(dats.shape)
                pass
            f_grid = get_freqs(freq,chunk=j)
            mean_f = np.mean(f_grid)
            avg_vel = (mean_f/freq-1)*1500
            axis.text(f_grid[0], dats[0,0], str(avg_vel)[:4])
            axis.scatter(mean_f, 0, color='r')
            power = np.sum(np.square(abs(dats)), axis=1)
            axis.plot(f_grid, power)
        if 'chunks' not in os.listdir('pics/'+str(freq)):
            os.system('mkdir pics/' + str(freq) +'/chunks')
        fig_name = 'pics/' + str(freq) + '/chunks/' + str(i).zfill(3) + '.png'
        plt.suptitle("Gamma = " + str(gamma))
        plt.savefig(fig_name)
        plt.close(fig)
    vid_name = 'chunk_stack.mp4'
    os.chdir('pics/' + str(freq) + '/chunks')
    os.system('ffmpeg -r 10 -f image2 -s 1920x1080 -i %03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ' + vid_name)
    os.chdir('../../..')
    return

def animate_figs(freq,chunk=None):
    vid_name = str(freq) + '.mp4'
    if type(chunk) != type(None):
        os.chdir('pics/gammapows/' + str(freq) + '/' + str(chunk))
    else:
        os.chdir('pics/gammapows/' + str(freq))
    os.system('ffmpeg -r 10 -f image2 -s 1920x1080 -i %03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ' + vid_name)
    os.chdir('../../../')
    if type(chunk) != type(None):
        os.chdir('..')

def limit_peaks(peak_inds, peak_heights, num_modes):
    """
    If there are too many peaks, only keep num_modes of the 
    highest ones
    peak_inds - np array
    peak_heights - np array
    num_modes - int
    """
    new_order = np.argsort(peak_heights)
    peak_inds = np.array([peak_inds[x] for x in new_order])
    peak_inds = peak_inds[::-1]
    peak_inds = peak_inds[:num_modes]
    peak_heights = np.array([peak_heights[x] for x in new_order])
    peak_heights = peak_heights[::-1]
    peak_heights = peak_heights[:num_modes]
    """ Now sort the peaks back from left to right """
    new_order = np.argsort(peak_inds)
    peak_inds = np.array([peak_inds[x] for x in new_order])
    peak_heights = np.array([peak_heights[x] for x in new_order])
    return peak_inds, peak_heights

def get_f_brackets(freqs, num_modes):
    """
    Idea is that modes should show up in
    difft parts of the spectrum
    Split up freqs into num_mode brackets
    """
    num_freqs = len(freqs)
    chunk_size = num_freqs//num_modes
    left_inds = [chunk_size*i for i in range(num_modes)]
    right_inds = [chunk_size*(i+1) for i in range(num_modes)]
    """Last one should just og to the end """
    right_inds[-1] = -1
    return list(zip(left_inds, right_inds))
    
def gen_alt_figs(freq, num_modes, chunk_len=1024, threshold=1000, short=False,chunk=None):
    """ 
    For each value of gamma, pick out the peaks from the spectrum
    For each peak, I get a corresponding mode shape
    Create a subplots fig for all the possible peaks...
    """
    num_rows, num_cols = get_subplot_shape(num_modes)
    fig,axes = plt.subplots(num_rows, num_cols)

    names = get_names(freq)
    gammas = get_gammas_from_names(names)
    print(gammas)
    depths=  get_depths()
    f = get_freqs(freq)
    start_cut=  250
    end_cut = -1
    lim_f = f[start_cut:end_cut] 
    ind_bracks = get_f_brackets(lim_f, num_modes) 
    for i, gamma in enumerate(gammas):
        fname = str(gamma)[:7]+'.npy'
        x = get_dats_for_gamma(gamma, freq, chunk_len, short=short,chunk=chunk)
        x = x[start_cut:end_cut,:]
        power = np.square(abs(x))
        if i == 0:
            psums = np.zeros((len(names), x.shape[0]))
            powers = np.zeros(power.shape)

        fig_name = 'pics/gammapows/' + str(freq) + '/' + str(i).zfill(3) + '.png'
        plt.suptitle('Gamma = ' +str(gamma))
        powers += np.sqrt(power)
        psum = np.sum(power, axis=1)
        psums[i,:] = psum[:]
        for j in range(len(ind_bracks)):
            inds = ind_bracks[j]
            left_ind, right_ind = inds[0], inds[1]
            peak_inds, peak_heights = find_peaks(psum[inds[0]:inds[1]], height=threshold)
            peak_heights = peak_heights['peak_heights']
            peak_ind = peak_inds[np.argmax(peak_heights)]
            if j%num_rows == 0:
                plt.xlim([-1.2, 1.2])
            subplot = num_rows*100 + (num_cols)*10 + j+1
            plt.subplot(subplot)
            mode_shape = x[peak_ind,:]
            mode_shape_r = mode_shape.real
            mode_shape_i = mode_shape.imag
            norm_fact = np.max(abs(mode_shape_r))
            norm_fact_i = np.max(abs(mode_shape_i))
            norm_fact = np.max([norm_fact, norm_fact_i])
            mode_shape_r /= norm_fact
            mode_shape_i /= norm_fact
            plt.plot(mode_shape_r,depths,color='b')
            plt.plot(mode_shape_i,depths, color='r')
        plt.savefig(fig_name)
        plt.clf()
    plt.show()

    

    plt.figure()
    plt.suptitle('Spectrum vs. gamma')
    levels = np.linspace(np.min(psums), np.max(psums), 30)
    plt.contour(f, gammas, psums, levels=levels)
    
    plt.figure()
    plt.suptitle('Summed pow')
    levels = np.linspace(np.min(powers), np.max(powers), 20)
    plt.contour(f, depths, powers.T, levels=levels)
    plt.show()

def make_mode_movie(freq, bin_ind,short=False,chunk=None):
    names = get_names(freq)
    gammas = get_gammas_from_names(names)
    depths = np.linspace(0, 63, 64)
    depths = np.delete(depths, 21)
    f = get_freqs(freq)
    pows = []
    xlim = 0
    for i, gamma in enumerate(gammas):
        x = get_dats_for_gamma(gamma, freq, chunk_len, short=short,chunk=chunk)
        x = x[bin_ind,:]
        if np.max(abs(x.real)) > xlim:
            xlim = np.max(abs(x.real))
        if np.max(abs(x.imag)) > xlim:
            xlim = np.max(abs(x.imag))
        pows.append(np.var(x))
    for i, gamma in enumerate(gammas):
        x = get_dats_for_gamma(gamma, freq, chunk_len, short=short,chunk=chunk)
        x = x[bin_ind,:]
        fig, (ax1, ax2, ax3)=plt.subplots(1,3)
        ax1.plot(x.real, depths)
        ax1.invert_yaxis()
        ax1.set_xlim([-xlim, xlim])
        ax2.plot(x.imag, depths)
        ax2.invert_yaxis()
        ax2.set_xlim([-xlim, xlim])
        ax3.plot(gammas, pows)
        ax3.scatter(gamma, pows[i], color='r')
        plt.suptitle('Value along array at frequency '+ str(f[bin_ind])+ ' for gamma = '+ str(gamma))
        plt.ylabel('Depth')
        ax1.set_xlabel('Real mode amp')
        ax2.set_xlabel('Imag mode amp')
        ax3.set_xlabel('Gamma')
        plt.savefig('pics/' + str(freq) + '/tmp/' + str(i).zfill(3))
        plt.close(fig)
    os.chdir('pics/' + str(freq) + '/tmp')
    vid_name = 'mode_sim_' + str(bin_ind).zfill(3) + '.mp4'
    os.system('ffmpeg -r 10 -f image2 -s 1920x1080 -i %03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ' + vid_name)
    os.system('rm *.png')
    os.chdir('../../../')

def get_subplot_shape(num_modes):
    """
    Get number of rows and number of cols
    to plot the modes
    """
    num_cols = int(np.sqrt(num_modes))
    num_rows = int(np.ceil(num_modes/num_cols))
    print(num_rows, num_cols)
    return num_rows, num_cols

def grab_mode_ests(freq, threshold, phi,chunk_len=1024, short=False,chunk=None):
    """
    Get the mode estimates from the demodulated
    data 
    """
    print(short, chunk)
    names = get_names(freq, chunk_len, short=short,chunk=chunk)
    f = get_freqs(freq)
    gammas = get_gammas_from_names(names, short=short,chunk=chunk)
    print(gammas)
    good_freqs = []
    good_inds = []
    depths = get_depths()
    good_gammas = []
    num_modes = phi.shape[1]
    num_rows, num_cols = get_subplot_shape(num_modes)
    fig, axes = plt.subplots(num_rows, num_cols)
    best_pows = [0]*num_modes
    best_gammas = [0]*num_modes
    best_freqs = [0]*num_modes
    best_finds = [0]*num_modes
    best_ests = [0]*num_modes
    """
    This way looks only at the real part or only the imaginary part 
    of the frequency correlations (aka it says it's a match if the 
    real part or the imaginary part matches)
    I suspect that the better mode shapes may not register as
    good matches because of some phase shifts along the array?
    Perhaps due to array tilt or something like that?
    """
    for i, gamma in enumerate(gammas):
        x = get_dats_for_gamma(gamma, freq, chunk_len,short=short,chunk=chunk)
        power = np.square(x)
        p_sum = np.sum(power, axis=1)
        #plt.plot(f, incoh_sum)
        peak_inds, peak_heights = find_peaks(p_sum.real, height=threshold)
        for j in peak_inds:
            mode_shape = x[j,:]
            mode_shape.reshape((mode_shape.size, 1))
            mode_shape_r = mode_shape.real
            mode_shape_r /= np.max(abs(mode_shape_r))
            mode_shape_r *= np.sign(mode_shape_r[0])
            best_ind, best_pow = get_best_match(mode_shape_r, phi)
            if best_pow > best_pows[best_ind]:
                best_pows[best_ind] = best_pow
                best_gammas[best_ind] = gamma
                best_freqs[best_ind] = f[j]
                best_finds[best_ind] = j
                curr_ax = axes[best_ind//num_cols, best_ind%num_cols]
                curr_ax.clear()
                curr_ax.plot(mode_shape_r, depths, color='b')
                best_ests[best_ind] = mode_shape_r
        peak_inds, peak_heights = find_peaks(p_sum.imag, height=threshold)
        for j in peak_inds:
            mode_shape = x[j,:]
            mode_shape.reshape((mode_shape.size, 1))
            mode_shape_i = mode_shape.imag
            mode_shape_i /= np.max(abs(mode_shape_i))
            mode_shape_i *= np.sign(mode_shape_i[0])
            best_ind, best_pow = get_best_match(mode_shape_i, phi)
            if best_pow > best_pows[best_ind]:
                best_pows[best_ind] = best_pow
                best_gammas[best_ind] = gamma
                best_freqs[best_ind] = f[j]
                best_finds[best_ind] = j
                curr_ax = axes[best_ind//num_cols, best_ind%num_cols]
                curr_ax.clear()
                curr_ax.plot(mode_shape_i, depths, color='b')
                best_ests[best_ind] = mode_shape_i
    for i in range(num_modes):
        ax = axes[i//num_cols, i%num_cols]
        ax.plot(phi[:, i], depths, color='r')
        ax.invert_yaxis() 

    """ Maybe implement an alternative here where I somehow fit the complex
    mode shapes to the modeled modes? But how..."""
    
    plt.suptitle(str(freq))
    #plt.show()
    dom = np.linspace(0, len(f) -1, len(f))

    #plt.plot(dom, f)
    print(best_freqs, best_finds)
    print(f[0], f[-1])
    #best_finds = [x for x in best_finds if x != 0]
    #best_freqs = [x for x in best_freqs if x != 0]
    #plt.scatter(best_finds, best_freqs, color='r')
    #plt.show()
    return best_gammas, best_finds, best_ests

def load_mode_shapes(freq):
    x = np.load('npy_files/' + str(freq) + '_shapes.npy')
    return x

def load_mode_krs(freq):
    x = np.load('npy_files/' + str(freq) + '_krs.npy')
    return x
    
def get_best_match(mode_shape, phi, method='power'):
    """
    Input 
    phi - np ndarray
        columns are modes
    mode_shape - np array
        single mode_shape
    """
    if method == 'power':
        power = mode_shape.T@phi
        best_ind = np.argmax(abs(power))
    return best_ind, power[best_ind]

def norm_phi(phi):
    """ normalize the mode-shapes"""
    for i in range(phi.shape[1]):
        phi[:, i] /= np.max(abs(phi[:,i]))
        phi[:,i] *= np.sign(phi[0,i])
    return phi

def get_ssp():
    """
    Get the sound speed on the array
    """
    env_builder  =factory.create('swellex')
    env = env_builder()
    zs = 10
    zr = np.linspace(94.125, 212.25,64)
    zr = np.delete(zr, 21)
    freq = 100
    env.add_source_params(100, zs, zr)
    print(env.z_ss.size, env.cw.size)
    z_ss = env.z_ss.reshape(env.z_ss.size)
    cw = env.cw.reshape(env.cw.size)
    c_func = interp1d(z_ss, cw)
    array_ssp = c_func(zr)
    return  array_ssp

def load_mode_model(freq):
    """ Get kraken mode shapes """
    phi = load_mode_shapes(freq) 
    phi = phi.real
    phi = norm_phi(phi)
    krs = load_mode_krs(freq)
    return phi, krs

def data_inverse():
    freqs = [49, 64, 79, 109, 127]
    freqs = [109]
    threshold = 100
    chunk_len=  1024
    mes= []
    for freq in freqs:
        phi = load_mode_shapes(freq)
        phi = phi.real
        phi = norm_phi(phi)
        best_gammas, best_finds, best_ests = grab_mode_ests(freq, threshold, phi, chunk_len=chunk_len)
        plt.show()
        print('best gammas', best_gammas)
        ests = [x for x in best_ests if type(x) != type(0)]
        err = []
        for i, est in enumerate(ests):
            diff = est - phi[:,i]
            err.append(np.var(diff))
        #plt.figure()
        me1 = ModeEstimates(freq, ests)
        me2 = ModeEstimates(freq, [phi[:,i] for i in range(phi.shape[1])])

        mes.append(me2)


    H = build_model_mat(mes)
    y = build_data_mat(mes)
    dats = np.linalg.inv(H.T@H)@H.T@y
    c = np.sqrt(1/dats[:63])
    plt.figure()
    plt.plot(dats[63:])
    plt.show()
    plt.figure()
    plt.plot(c)
    plt.show()
        
    names = get_names(freq)
       
    f = get_freqs(freq) 
    print(np.mean(f))
    best_freqs = [f[i] for i in best_finds if i != 0]
    Delta_f = f[-1] - f[0]
    print("DETA F", Delta_f)
    print(best_freqs - np.mean(f))

def testing_sine_hypoth(freq):
    """
    Just see if we can get reasonable approximations
    to the normalized mode shapes using a modulated
    kz using known kr and ssp """
    om = 2*np.pi*freq
    array_ssp = get_ssp()
    phi, krs = load_mode_model(freq)
    zr = np.linspace(94.125, 212.25,64)
    zr = np.delete(zr, 21)
    for i in range(len(krs)):
        kz = np.sqrt(om*om/np.square(array_ssp) - krs[i]*krs[i])
        mod_shape = phi[:,i]
        p0 = np.arccos(mod_shape[0])
        p0 -= kz[0]*zr[0]
        test_shape = np.cos(p0 + kz*zr)
        plt.figure()
        plt.plot(zr, test_shape)
        plt.plot(zr, mod_shape)
        plt.gca().invert_yaxis()
        plt.show()

def testing_inverse(freq):
    om = 2*np.pi*freq
    array_ssp = get_ssp()
    phi, krs = load_mode_model(freq)
    zr = np.linspace(94.125, 212.25,64)
    zr = np.delete(zr, 21)
    plt.figure()
    plt.plot(zr, array_ssp)
    plt.gca().invert_yaxis()
    me = ModeEstimates(freq, [phi[:,i] for i in range(len(krs))])
    plt.figure()
    for x in me.phis:
        plt.plot(zr, x)
    plt.figure()
    for i in range(krs.size):
        plt.plot(zr, phi[:,i])
    plt.show()

def get_om_s():
    """ Try and estimate omega_s using
    equation 38 in the Walker paper 
    Start out by just plotting gamma_m and omega_m"""
    freqs = [49, 64, 79, 94, 109, 127]
    threshold = 500
    chunk_len=  1024
    short=False
    freqs = [127, 385]
    chunk=0

    mes= []
    for freq in freqs:
        f = get_freqs(freq)
        phi, krs = load_mode_model(freq)
        phi = phi[1:,:]
        phi = phi.real
        phi = norm_phi(phi)
        best_gammas, best_finds, best_ests = grab_mode_ests(freq, threshold, phi, chunk_len=chunk_len,short=short,chunk=chunk)
        print('freq', best_gammas)
        fig,ax = plt.subplots(1,1)
        fig.suptitle('Peak frequency versus mode match for ' + str(freq))
        ax.plot(krs, f[best_finds])
        print(krs)
        plt.show()
        
        ests = [x for x in best_ests if type(x) != type(0)]
        err = []
        for i, est in enumerate(ests):
            diff = est - phi[:,i]
            err.append(np.var(diff))
        #plt.figure()
        me1 = ModeEstimates(freq, ests)
        me1.build_mode_mat()
        mat = me1.mode_mat
        np.save('npy_files/mode_mat_' + str(freq) + '.npy', mat)
        print(mat.shape)
        tmp =  np.linalg.pinv(mat)
        print(tmp.shape)

def get_spec_gammas(freq):
    fs = os.listdir('./npy_files/' + str(freq) + '/spec')
    mode_mat = np.load('npy_files/mode_mat_' + str(freq) + '.npy')
    print(mode_mat.shape)
    freqs = get_freqs(freq)
    gammas = []
    gstrings=[]
    zr = np.linspace(94.125, 212.25, 64)
    zr = np.delete(zr, 21)
    for f in fs:
        i = 0
        while f[i] != '_':
            i+=1
        gstring = f[:i]
        gamma = float(gstring)
        gstrings.append(gstring)
        gammas.append(gamma)

    
    fname = './npy_files/'+str(freq) + '/spec/' + gstring + '_spec_1024.npy'
    x = np.load(fname)
    x = x.T
    num_modes = x.shape[0]
            
    inds = np.argsort(gammas)
    gstrings_sorted = [gstrings[x] for x in inds]


    for mode_num in range(num_modes):
        for i, gstring in enumerate(gstrings_sorted):
            fname = './npy_files/'+str(freq) + '/spec/' + gstring + '_spec_1024.npy'
            x = np.load(fname)
            x=x.T
            fig,axes = plt.subplots(1,2)
            axes[0].set_ylim([0, 4500])
            axes[0].plot(freqs, np.square(abs(x[mode_num,:])))
            plt.xlim([-1.4, 1.4])
            axes[1].set_xlim([-1.2, 1.2])
            axes[1].plot(mode_mat[:,mode_num],zr)
            plt.suptitle('gamma = '+ gstring)
            plt.savefig('./pics/64/spec/' + str(i).zfill(3) + '.png')
            plt.close(fig)
                
        os.chdir('pics/' + str(freq) + '/spec')
        vid_name = str(mode_num) + '.mp4'
        os.system('ffmpeg -r 10 -f image2 -s 1920x1080 -i %03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ' + vid_name)
        os.system('rm *.png')
        os.chdir('../../../')
        
def plot_mode_spec():
    freqs = [49]
    for f in freqs:
        x = np.load('npy_files/' + str(f) + '/0.0_spec_1024.npy')
        x = x.T
        for i in range(x.shape[0]):
            plt.plot(abs(x[i,:]))
        plt.legend([str(i) for i in range(x.shape[0])])
        plt.show()

def compare_mode_spec():
    freq = 127
    root = './npy_files/'+str(freq) + '/filt/'
    fs = os.listdir(root)
    for i,f in enumerate(fs):
        x = np.load(root+f)
        x = x.T
        plt.subplot(511+i)
        plt.plot(x[1,:])
    plt.show()
        
   
if __name__ == '__main__': 
       
    #get_om_s()
    for i in range(1): 
        gen_figs(109, short=False,chunk=i)
        animate_figs(109, chunk=i)
