import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import pickle
from scipy.signal import detrend, convolve, firwin, hilbert
#import matplotlib._color_data as mcd
#from lse import get_brackets
#from matplotlib.widgets import PolygonSelector
#from matplotlib.path import Path
from swellex.audio.brackets import Bracket, get_brackets, form_good_pest, form_good_alpha,get_pest
from swellex.audio.kay import get_pest_fname
from swellex.audio.config import get_proj_root

'''
Description:
Process the phase estimates produces by kay.py

Date: 

Author: Hunter Akins
'''

def lin_reg_chunk(p_est, T, dt, overlap=.5):
    """
    Break a phase estimate array into chunks
    Perform linear regression on each chunk
    The slope can be used as a freq. estimate
    Which is a proxy for velocity
    Input 
    p_est - numpy 1d array
        phase estimates (radians)
    T - float 
        Chunk size (s)
    dt - float
        Time step (for converting interval to indices)
    overlap - float
        percentage overlap between regression window estimates
    Output -
    omega_ests - np array
        estimates slope of phsae line
    errs - np array 
        errors in linear regression
    """
    """ Compute num samples per chunk """
    num_sam =int( T // dt )
    """ Compute number of chunks """
    N = p_est.size
    step = int(num_sam * (1-overlap))
    """ ignores a bit at the end which is fine for small intervals"""
    num_ch =int( (N-step) // step)  
    """ Compute domain of sampling """
    dom = np.linspace(0, N, N)
    omega_ests = np.zeros(num_ch)
    errs = np.zeros(N)
    t = np.linspace(0, num_sam, num_sam)
    H = t.reshape(t.size, 1)
    pinv = np.linalg.inv(H.T@H)@H.T
    err_vars = np.zeros(num_ch)
    """ loop over chunks and perform the regression  """
    for ch in range(num_ch):
        inds = slice(ch*step, ch*step+num_sam)
        phases = p_est[inds]
        phases = phases-phases[0]
#        t = dom[inds]
        omega_est = pinv@phases
        omega_ests[ch] = omega_est
        err = omega_est*t - phases
        err_vars[ch] = np.var(err)
        errs[inds] = omega_est*t - phases
    return omega_ests, errs, err_vars

def spec_reg_chunk(p_est, T, dt, num_f, overlap=.5):
    """
    Break a phase estimate array into chunks
    Perform linear regression on each chunk
    The slope can be used as a freq. estimate
    Which is a proxy for velocity
    Input 
    p_est - numpy 1d array
        phase estimates (radians)
    T - float 
        Chunk size (s)
    dt - float
        Time step (for converting interval to indices)
    overlap - float
        percentage overlap between regression window estimates
    Output -
    omega_ests - np array
        estimates slope of phsae line
    errs - np array 
        errors in linear regression
    """
    """ Compute num samples per chunk """
    num_sam =int( T // dt )
    """ Compute number of chunks """
    N = p_est.size
    step = int(num_sam * (1-overlap))
    """ ignores a bit at the end which is fine for small intervals"""
    num_ch =int( (N-step) // step)  
    """ Compute domain of sampling """
    dom = np.linspace(0, N, N)
    errs = np.zeros(N)
    """ Compute H as a sum of frequency """
    t = np.linspace(0, num_sam, num_sam)
    H = np.zeros((num_sam, 2*num_f+1))
    H[:,0] = t
    if num_f > 0:
        df = 1/num_sam
        for i in range(1, num_f+1):
            vals = np.cos(2*np.pi*i*df*t)
            H[:,2*i-1] = vals
            vals = np.sin(2*np.pi*i*df*t)
            H[:,2*i] = vals
    prod = (H.T)@H
    u,s,vh=np.linalg.svd(prod)
    pinv = np.linalg.inv(H.T@H)@H.T
    omega_ests = np.zeros((num_ch, 2*num_f+1))
    """ loop over chunks and perform the regression  """
    for ch in range(num_ch):
        inds = slice(ch*step, ch*step+num_sam)
        phases = p_est[inds]
        phases = phases-phases[0]
#        t = dom[inds]
        omega_est = pinv@phases
        omega_est = omega_est
        omega_ests[ch,:] = omega_est
        err = H@omega_est - phases
        win_vals = np.hamming(err.size)
        errs[inds] = err
    return omega_ests, errs

def poly_reg_chunk(p_est, T, dt, num_p, overlap=.5):
    """
    Break a phase estimate array into chunks
    Perform linear regression on each chunk
    Using a polynomial of degree num_p
    Input 
    p_est - numpy 1d array
        phase estimates (radians)
    T - float 
        Chunk size (s)
    dt - float
        Time step (for converting interval to indices)
    num_p - int 
        Polynomial degree -1 to use
    overlap - float
        percentage overlap between regression window estimates
    Output -
    omega_ests - np array
        estimates slope of phsae line
    errs - np array 
        errors in linear regression
    """
    """ Compute num samples per chunk """
    num_sam =int( T // dt )
    """ Compute number of chunks """
    N = p_est.size
    step = int(num_sam * (1-overlap))
    """ ignores a bit at the end which is fine for small intervals"""
    num_ch =int( (N-step) // step)  
    """ Compute domain of sampling """
    dom = np.linspace(0, N, N)
    omega_ests = np.zeros((num_ch, num_p))
    errs = np.zeros(N)
    t = np.linspace(0, num_sam, num_sam)
    t = t - num_sam // 2
    H = np.zeros((t.size, num_p))
    for i in range(num_p):
        power = i+1
        tval = np.power(t, power)
        H[:,i] = tval
        plt.plot(H[:,i])
        plt.show()
    pinv = np.linalg.inv(H.T@H)@H.T
    """ loop over chunks and perform the regression  """
    for ch in range(num_ch):
        inds = slice(ch*step, ch*step+num_sam)
        phases = p_est[inds]
        phases = phases-phases[0]
#        t = dom[inds]
        omega_est = pinv@phases
        print(omega_est)
        omega_ests[ch,:] = omega_est
        errs[inds] = H@omega_est - phases
    return omega_ests, errs

def lump_lin_reg(p_est, T, dt, overlap=.5):
    """
    Break a phase estimate array into chunks
    Perform linear regression on each chunk
    The slope can be used as a freq. estimate
    Which is a proxy for velocity
    Input 
    p_est - numpy 2d array
        phase estimates (radians) are as columns
    T - float 
        Chunk size (s)
    dt - float
        Time step (for converting interval to indices)
    overlap - float
        percentage overlap between regression window estimates
    Output -
    omega_ests - np array
        estimates slope of phsae line
    errs - np array 
        errors in linear regression
    """
    """ Compute num samples per chunk """
    num_sam =int( T // dt )
    """ Compute number of chunks """
    N = p_est.shape[0]
    step = int(num_sam * (1-overlap))
    """ ignores a bit at the end which is fine for small intervals"""
    num_ch =int( (N-step) // step)  
    """ Compute domain of sampling """
    dom = np.linspace(0, N, N)
    omega_ests = np.zeros(num_ch)
    """ Compute number of sensors and form linear model matrix """
    num_sensors = p_est.shape[1]
    errs = np.zeros((num_ch*num_sam, num_sensors))
    t = np.linspace(0, num_sam, num_sam)
    H = np.zeros(num_sam*num_sensors)
    for i in range(num_sensors):
        indices = slice(i*num_sam, (i+1)*num_sam)
        H[indices] = t
    H = H.reshape(H.size, 1)
    pinv = np.linalg.inv(H.T@H)@H.T
    """ loop over chunks and perform the regression  """
    for ch in range(num_ch):
        inds = slice(ch*step, ch*step+num_sam)
        phases = p_est[inds,:]
        phases = phases-phases[0,:]
        x=phases[:,0]
        col_ph = phases.T.reshape(H.size)
#        t = dom[inds]
        omega_est = pinv@col_ph
        omega_ests[ch] = omega_est
        est_phase = omega_est*H
        err = est_phase[:,0] - col_ph
        err = err.reshape(num_sam, num_sensors)
        errs[inds,:] =err
    return omega_ests, errs
        
def data_run(freqs, sensor_inds, interval):
    """
    Process the pickled phase estimates from the data in data_run in kay
    Input - 
    freqs - list
    list of frequencies
    sensor_inds - list
    list of sensor inds
    interval - float
    Length of the chunk of time over which I perform a linear regression
    For example, interval = 10 means I gather 10 second chunks 
    of phase estimates
    and do a least squares fit on each of them with no overlap
    """
    fs = 1500
    dt = 1/fs
    for f0 in freqs:
        for s_ind in sensor_inds:
            """ Load the phsae estimate """
            fn = f0*dt # convert to 1 Hz sampling
            p_est = get_pest(f0, s_ind)
            """ Get lenght of estimate, set up the domain """
            N = p_est.size
            domain = np.linspace(0,N,N)
            guess_vals = 2*np.pi*fn*domain
            """ Loop over the estimates and 
                estimate the slope for each chunk """
            omega_ests,errs = lin_reg_chunk(p_est, interval, dt)
            fn = f0*dt
            fn_ests = omega_ests/2/np.pi
            vbar = 1500*((fn_ests /fn)-1) # average vel 1+v/1500
            f_ests = fn_ests/dt 

def compare_sensor_phase(freqs, s_inds):
    """
    Compare the phase estimates for each sensor in the array
    Input 
    freqs - list
        frequencies of interest
    Output 
    saves a nice figure
    maybe looks at some stats?
    """
    for f0 in freqs:
        fig =plt.figure()
        plt.suptitle('Difference in phase estimates derived from ith sensor \n  to estimates from the first sensor on the array \nfor the ' + str(f0) + ' Hz band')
        ref_est = get_pest(f0, 1)
        dt = 1/1500
        dom = dt*np.linspace(0, ref_est.size, ref_est.size)
        for s_ind in s_inds:
            p_est = get_pest(f0, s_ind)
            diff = ref_est - p_est
            plt.plot(dom,diff)
        plt.ylabel('Phase difference (radians)')
        plt.legend([str(x) for x in s_inds], loc='upper right')
        plt.xlabel('Time (s)')
        plt.savefig('pics/phases/'+str(f0) + '_simple_diff.png')
        plt.clf()
        plt.close(fig)
        
def plot_detrended_phase(freqs, s_inds):
    """
    pretty suggestive name..
    Input 
    freqs - list
        frequencies of interest
    Output 
    """
    for f0 in freqs:
        fig =plt.figure()
        plt.suptitle('Detrended phase estimates in the  ' + str(f0) + ' Hz band')
        ref_est = get_pest(f0, 1)
        dt = 1/1500
        dom = dt*np.linspace(0, ref_est.size, ref_est.size)
        for s_ind in s_inds:
            p_est = get_pest(f0, s_ind)
            p_est = detrend(p_est)
            plt.plot(dom,p_est)
        plt.ylabel('Phase difference (radians)')
        plt.legend([str(x) for x in s_inds], loc='upper left')
        plt.xlabel('Time (s)')
        plt.savefig('pics/phases/'+str(f0) + '_detrended.png')
        #plt.clf()
        #plt.close(fig)
    
def plot_lin_regs(freqs, s_inds, interval, overlap=.5):
    """
    Perform linear regression on chunks of estimates
    of length interval (s)
    Input - 
    freqs - list
    list of frequencies
    sensor_inds - list
    list of sensor inds
    interval - float
    Length of the chunk of time over which I perform a linear regression
    """
    fs = 1500
    dt = 1/fs
    for f0 in freqs:
        fig = plt.figure()
        plt.suptitle('Estimate of linear parameter at ' + str(f0) + ' Hz\n for windows of ' +str(interval) + ' seconds and overlap of ' + str(overlap))
        for s_ind in sensor_inds:
            """ Load the phsae estimate """
            p_est = get_pest(f0, s_ind)
            """ Get lenght of estimate, set up the domain """
            N = p_est.size
            domain = np.linspace(0,N,N)
            """ Loop over the estimates and 
                estimate the slope for each chunk """
            err_var = []
            omega_ests,errs = spec_reg_chunk(p_est, interval, dt, 0, overlap)
            f_ests = omega_ests/2/np.pi/dt
            plt.plot(f_ests)
#        plt.legend([str(x) for x in sensor_inds], loc='upper left')
        plt.ylabel('Estimated slope (Hz)')
        plt.xlabel('Window (time)')
        plt.savefig('pics/phases/' + str(f0) + 'raw_reg.png')
        return omega_ests
        #plt.close(fig)

def plot_order_effect(freqs, s_inds, interval, overlap=.5):
    """
    Perform linear regression on chunks of estimates
    of length interval (s)
    Input - 
    freqs - list
    list of frequencies
    sensor_inds - list
    list of sensor inds
    interval - float
    Length of the chunk of time over which I perform a linear regression
    """
    fs = 1500
    dt = 1/fs
    for f0 in freqs:
        fig = plt.figure()
        plt.suptitle('Error variance as a function of number of included frequencies (model error) at ' + str(49) + ' Hz')
        for s_ind in sensor_inds:
            """ Load the phsae estimate """
            p_est = get_pest(f0, s_ind)
            """ Get lenght of estimate, set up the domain """
            N = p_est.size
            domain = np.linspace(0,N,N)
            """ Loop over the estimates and 
                estimate the slope for each chunk """
            err_var = []
            for j in range(7):
                num_f = j*5
                omega_ests,errs = spec_reg_chunk(p_est, interval, dt, num_f, overlap)
                lins = omega_ests[:,0]
                fil = firwin(100, .1)
                lpom = convolve(lins, fil, mode='same')
                plt.plot(omega_ests[:,0])
                plt.plot(lpom)
                plt.show()
                err_var.append(np.var(errs))
            f_ests = omega_ests/2/np.pi/dt
            plt.plot(err_var)
        #plt.savefig('pics/phases/' + str(f0) + 'order_comp.png')
#        plt.legend([str(x) for x in sensor_inds], loc='upper left')
        plt.ylabel('Error var')
        plt.xlabel('Model order')
        plt.savefig('pics/phases/' + str(f0) + 'raw_reg.png')
        #plt.close(fig)

def plot_lin_reg_errs(freqs, s_inds, interval, overlap=.5):
    """
    Perform linear regression on chunks of estimates
    of length interval (s)
    Plot errors
    Input - 
    freqs - list
        list of frequencies
    sensor_inds - list
        list of sensor inds
    interval - float
        Length of the chunk of time over which I perform a linear regression
    """
    fs = 1500
    dt = 1/fs
    for f0 in freqs:
        fig = plt.figure()
        plt.suptitle('Error in regression line for f= ' + str(f0) +' Hz\n')
        for s_ind in sensor_inds:
            """ Load the phsae estimate """
            p_est = get_pest(f0, s_ind)
            """ Get lenght of estimate, set up the domain """
            N = p_est.size
            domain = np.linspace(0,N,N)
            """ Loop over the estimates and 
                estimate the slope for each chunk """
            omega_ests,errs = lin_reg_chunk(p_est, interval, dt, overlap)
            f_ests = omega_ests/2/np.pi/dt
            plt.plot(errs[:interval*fs])
            plt.show()
        plt.legend([str(x) for x in sensor_inds], loc='upper left')
        plt.ylabel('Err (radians)')
        plt.xlabel('Window number')
        plt.savefig('pics/phases/' + str(f0) + 'reg_errs.png')
        plt.close(fig)

def plot_lump_regs(freqs, s_inds, interval, overlap=.5):
    """
    Perform linear regression on chunks of estimates
    of length interval (s)
    Input - 
    freqs - list
    list of frequencies
    sensor_inds - list
    list of sensor inds
    interval - float
    Length of the chunk of time over which I perform a linear regression
    """
    fs = 1500
    dt = 1/fs
    num_sensors = len(s_inds)
    for f0 in freqs:
        fig = plt.figure()
        plt.suptitle('Sliding linear regression for f= ' + str(f0) +' Hz\nInterval of window is ' + str(interval) + 's with overlap ' + str(overlap))
        for s_ind in sensor_inds:
            """ Load the phsae estimate """
            p_est = get_pest(f0, s_ind)
            if s_ind == 1:
                p_ests = np.zeros((p_est.size, num_sensors))
            p_ests[:, s_ind-1] = p_est
        """ Get lenght of estimate, set up the domain """
        """ Loop over the estimates and 
            estimate the slope for each chunk """
        omega_ests,errs = lump_lin_reg(p_ests, interval, dt, overlap)
        f_ests = omega_ests/2/np.pi/dt
        plt.plot(f_ests)
        plt.ylabel('Regression slope (Hz)')
        plt.xlabel('Window number')
        plt.savefig('pics/phases/' + str(f0) + 'lump_reg.png')
        plt.show()
        plt.close(fig)
        fig = plt.figure()
        plt.plot(errs[:,0])
        plt.show()

def compare_to_signal():
    interval = 1

    sensor = np.load('npy_files/sensor' + str(1) + '.npy') # zero mean
    sensor -= np.mean(sensor)
    sensor = sensor/np.sqrt(np.var(sensor))
    filt = firwin(1024, [49-.5, 49+.5], fs=1500, pass_zero=False, window='hann')
    sensor = convolve(sensor[:,0], filt, mode='same')
    """get the lin reg """
    N = interval*1500 
    sensor = sensor[:N]
    pest =get_pest(49, 1)
    dt = 1/1500
    overlap = 0
    omega_ests,errs = lin_reg_chunk(pest, interval, dt, overlap)
    pest = pest[:N]
    lin_p = pest[0] + np.linspace(0, N, N)*omega_ests[0]
    
    fig = plt.figure()
    plt.suptitle('Phase estimate versus data')
    ax = plt.subplot(211)
    T = np.linspace(0, N//2, N//2)/1500
    plt.plot(T, np.cos(pest[:N//2])/2)
    plt.plot(T, sensor[:N//2])
    plt.legend(['Estimated', 'Data'])
    ax.set_xticklabels([])
    plt.subplot(212)
    plt.title('Linear phase model versus data')
    plt.plot(T, np.cos(lin_p[:N//2])/2)
    plt.plot(T, sensor[:N//2])
    plt.xlabel('Time (s)')
    plt.savefig('pics/phases/49_lin_comp')
    plt.close(fig)
    
def plot_poly_regs(freqs, s_inds, interval, num_p, overlap=.5):
    """
    Perform linear regression on chunks of estimates
    of length interval (s) with a polynomial model
    Input - 
    freqs - list
    list of frequencies
    sensor_inds - list
    list of sensor inds
    interval - float
    Length of the chunk of time over which I perform a linear regression
    """
    fs = 1500
    dt = 1/fs
    for f0 in freqs:
        fig = plt.figure()
        plt.suptitle('Sliding linear regression for f= ' + str(f0) +' Hz\nInterval of window is ' + str(interval) + 's with overlap ' + str(overlap))
        for s_ind in sensor_inds:
            """ Load the phsae estimate """
            p_est = get_pest(f0, s_ind)
            """ Get lenght of estimate, set up the domain """
            N = p_est.size
            domain = np.linspace(0,N,N)
            """ Loop over the estimates and 
                estimate the slope for each chunk """
            omega_ests,errs = poly_reg_chunk(p_est, interval, dt, num_p, overlap)
            plt.plot(omega_ests)
            plt.show()
        plt.legend([str(x) for x in sensor_inds], loc='upper left')
        plt.ylabel('Regression slope (Hz)')
        plt.xlabel('Window number')
        plt.savefig('pics/phases/' + str(f0) + 'raw_reg.png')
        plt.show()
        plt.close(fig)

def compare_to_sig_amp(f):
    sensor = np.load('npy_files/sensor' + str(1) + '.npy') # zero mean
    sensor -= np.mean(sensor)
    sensor = sensor/np.sqrt(np.var(sensor))
    filt = firwin(1024, [f-.5, f+.5], fs=1500, pass_zero=False, window='hann')
    sensor = convolve(sensor[:,0], filt, mode='same')
    """get the lin reg """
    pest =get_pest(f, 1)
    fig = plt.figure()
    plt.suptitle('Comparing phase estimates to filtered data at ' + str(f) + ' Hz')
    plt.subplot(211)
    plt.plot(sensor)
    plt.subplot(212)
    plt.plot(detrend(pest))
    plt.savefig('pics/phases/sig_amp/'+ str(f) + '_sig_amp.png')
    plt.close(fig)

def compare_to_snr(f, sensor_ind):
    snr = np.load('npy_files/snr' + str(sensor_ind) + '.npy')
    snr_fs = np.load('npy_files/snr_freqs.npy')
    rel_ind = np.argmin(np.array([abs(f-x) for x in snr_fs]))
    print(f, snr_fs[rel_ind])
    snr = snr[rel_ind,:]
    pest =get_pest(f, sensor_ind)
    dom = np.linspace(6.5, 40, pest.size)
    snr_dom = np.linspace(6.5, 40, snr.size)
    fig = plt.figure()
    plt.suptitle('Comparing phase estimates to snr estimates at ' + str(f) + ' Hz')
#    plt.subplot(211)
#    plt.plot(snr_dom, snr)
#    plt.subplot(212)
    plt.plot(dom, detrend(pest))
    plt.savefig('pics/phases/sig_amp/'+ str(f) + '_sig_snr.png')
    plt.show()
    plt.close(fig)

def plot_filtered_pest(freqs, mins, sensor_ind=1):
    color_count = 0
    overlap = {name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS}
    overlap = sorted(overlap)
    plt.figure()
    handles = []
    labels = []
    plt.suptitle('High SNR portions of unwrapped phase for all signal bands in SwellEx 96')
    for f, brackets in zip(freqs,mins):
        col = mcd.XKCD_COLORS["xkcd:"+overlap[color_count]].upper()
        color_count += 2
        pest =get_pest(f, sensor_ind)
        dom = np.linspace(6.5, 40, pest.size)
        dt = dom[1]-dom[0]
        inds = []
        size = 0
        x,y = form_good_pest(f, brackets,sensor_ind)
        for dom_seg, pest_seg in zip(x,y):
            line = (plt.plot(dom_seg, pest_seg, color=col, label=str(f)))[0]
        handles.append(line)
        labels.append(str(f))
    plt.legend(handles, labels,loc='upper left')
    plt.xlabel('Time (min)')
    plt.ylabel('Unwrapped phase (radians)')
    plt.show()

def plot_lin_reg_good(freqs, mins, sensor_ind):
    color_count = 0
    overlap = {name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS}
    overlap = sorted(overlap)
    handles = []
    labels = []
    for f, brackets in zip(freqs,mins):
        col = mcd.XKCD_COLORS["xkcd:"+overlap[color_count]].upper()
        color_count += 2
        pest =get_pest(f, sensor_ind)
        dom = 60*np.linspace(6.5, 40, pest.size)
        dt = dom[1]-dom[0]
        inds = []
        size = 0
        dom_segs,pest_segs = form_good_pest(f, brackets,sensor_ind)
        """ Create obj to hold lin reg ests """
        fslopes = []
        for x,y in zip(dom_segs, pest_segs):
            x0 = x[0]
            x -= x[0]
            H = x.reshape(x.size, 1)
            pinv = np.linalg.inv(H.T@H)@H.T
            slope = pinv@(y-y[0])
            y_est = H@slope + y[0]
            v_est = 1500*(slope / 2 /np.pi / f/60 -1)
            y_err = y_est - y
            err_var = np.var(y_err)/np.sqrt(y_err.size)
            v_var = np.square(1500/2/np.pi/f/60)*err_var
            dom = x+x0
#            x_vals = [dom[0], (dom[0]+dom[-1])/2, dom[-1]]
            x_vals = [(dom[0]+dom[-1]) / 2]
            # line = plt.errorbar(x_vals, [v_est], [2*np.sqrt(v_var)], color=col,label=str(f))[0]
            line = plt.scatter(x_vals, [v_est], s=4, color=col,label=str(f))
        handles.append(line)
        labels.append(str(f))
    plt.ylim([2.2,3])
    plt.legend(handles, labels,loc='upper left')

def plot_resid_good(freq,brackets, sensor_ind):
    pest =get_pest(f, sensor_ind)
    dom = 60*np.linspace(6.5, 40, pest.size)
    dt = dom[1]-dom[0]
    inds = []
    size = 0
    dom_segs,pest_segs = form_good_pest(f, brackets,sensor_ind)
    for x,y in zip(dom_segs, pest_segs):
        x0 = x[0]
        x -= x[0]
        H = x.reshape(x.size, 1)
        pinv = np.linalg.inv(H.T@H)@H.T
        slope = pinv@(y-y[0])
        y_est = H@slope + y[0]
        v_est = 1500*(slope / 2 /np.pi / f/60 -1)
        y_err = y_est - y
        err_var =np.var(y_err)
        v_var = np.square(1500/2/np.pi/f/60)*err_var
        plt.plot(x+x0, y_err)
        plt.show()

def check_filter(f, brackets, sensor_ind=1):
    pest =get_pest(f, sensor_ind)
    dom = np.linspace(6.5, 40, pest.size)
    dt = dom[1]-dom[0]
    inds = []
    size = 0
    fig = plt.figure()
    x,y = form_good_pest(f, brackets,sensor_ind,detrend_flag=True)
    for dom_seg, pest_seg in zip(x,y):
        plt.plot(dom_seg, pest_seg)
    plt.xlabel('Time (min)')
    plt.ylabel('Unwrapped phase (radians)')
    plt.show()
    plt.close(fig)
    return

class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []
        self.inds = []
        self.mins = []

    def onselect(self, verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.xys))[0]
        self.ind = ind
        self.inds.append([ind[0], ind[-1]])
#        self.fc[:, -1] = self.alpha_other
#        self.fc[self.ind, -1] = 1
#        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

def select_inds(sensor):
    """ leave the dom alone it's off but I deal with it later in Brackets """
    freqs= [49, 64, 79, 94, 109, 112,127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    for f in freqs:
        print('-------------- ' + str(f) + ' hz')
        fig, ax = plt.subplots()
        pest =get_pest(f, sensor)
        dom = np.linspace(6.5, 40, pest.size)
        fig.suptitle(str(f) + ' Hz, sensor ' + str(sensor))
#       # compare_to_snr(f, 3)
        pts = ax.scatter(dom[::100], detrend(pest[::100]), s=1)
        selector = SelectFromCollection(ax, pts)
        plt.show()
        selector.disconnect()
        for x in selector.inds:
            print('[' + str(dom[100*x[0]]) + ', ' + str(dom[100*x[1]]) + '],')
   
def make_5s_bracks(pest):
    """
    Take in a pest, which has sampling rate 1500 Hz
    Get the full size of it
    Then return a list of lists
    Each sublist has a start_ind and end_ind
    corresponding to sequential 5s chunks of
    the domain
    """
    inds = np.linspace(0, pest.size-1, pest.size, dtype=int)
    fs = 1500
    num_samples = 5 *1500 # num samples in a 5s chunk
    num_bracks = pest.size // num_samples 
    bracks = []
    for i in range(num_bracks):
        start_ind = i*num_samples
        end_ind = (i+1)*num_samples
        bracks.append([start_ind, end_ind])
    return bracks
    
def auto_select_inds(sensor,cpa=False):
    """ 
    Develop an automatic detector for the good sections of phase
    data 
    Basic idea will be to look at variance of detrended 5s chunks
    """
    freqs= [49, 64, 79, 94, 109, 112,127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    if cpa==True:
        pest =get_cpa_pest(49, sensor)
    else: 
        pest = get_pest(49, sensor)
    bracks = make_5s_bracks(pest)
    num_bracks = len(bracks)
    good_bracks = []
    print('selecting good indices for sensor ', sensor)
    for f in freqs:
        brack_indicator = np.zeros(num_bracks, dtype=bool)
        f_good_bracks = []
        print('-------------- ' + str(f) + ' hz')
        if cpa == True:
            pest =get_cpa_pest(f, sensor)
        else: 
            pest = get_pest(f, sensor)
        detrend(pest, overwrite_data=True)
        for i in range(num_bracks):
            brack = bracks[i]
            data = pest[brack[0]:brack[1]]
            vals = detrend(data)
            var = np.var(vals)
            """ avoid the negative slope ones"""
            if var < .5:
                f_good_bracks.append(brack)
        good_bracks.append(f_good_bracks)
    return good_bracks

def auto_write_indices(fname, cpa=False):
    """
    Use the automatic bracket detector
    to create a python file with the mins object  
    for all sensors defined """
    with open(fname, 'w') as newfile:
        newfile.write('mins = [')
        for sensor in range(1,22):
            newfile.write('[')
            good_bracks = auto_select_inds(sensor, cpa)
            for i in range(len(freqs)):
                newfile.write('[')
                f = freqs[i]
                bracks = good_bracks[i]
                if cpa == True:
                    dats = get_cpa_pest(f, 1)
                else:
                    dats = get_pest(f,1)
                dats = detrend(dats)
                if bracks == []:
                    newfile.write('[]')
                else:
                    for j in range(len(bracks)-1):
                        brack = bracks[j]
                        vals = dats[brack[0]:brack[1]]
                        newfile.write('['+str(brack[0])+','+str(brack[1]) + '],')
                    brack = bracks[-1]
                    if i == (len(freqs)-1):
                        newfile.write('['+str(brack[0])+','+str(brack[1]) + ']]')
                    else:
                        newfile.write('['+str(brack[0])+','+str(brack[1]) + ']],\n')
                   # mini_dom = np.linspace(brack[0], brack[1], vals.size, endpoint=False)
            newfile.write('],\n')
        newfile.write(']')
    return

def get_bracket_name(freq, proj_string='s5'):
    proj_root = get_proj_root(proj_string)
    bname = proj_root + 'good_brackets_' + str(freq) + '.pickle'
    return bname

def auto_gen_brackets(freqs, chunk_size, var_lim=.5, proj_string='s5'):
    """Generate the brackets for 
    the full set of sensors , frequencies 
    For each frequency, load up the phase estimates
    then loop through each sensor and check...it might take a while
    input 
    freqs - list of ints
        source frequencies for which to identify the 
        good brackets
    chunk_size - int
        size of data chunks to consider (number of samples)
    """
    for f_ind in range(len(freqs)):
        good_bracks = []
        f=  freqs[f_ind]
        print('Processing frequency ', f)
        data_loc = get_pest_fname(f, proj_string=proj_string)
        print('Loading phase estimates from ', data_loc)
        pests = np.load(data_loc)
        total_samples = pests.shape[1]
        print('Number of time smaples', total_samples, total_samples/60/1500)
        num_brackets = int(total_samples // chunk_size)-1
        for j in range(num_brackets):
            start = j*chunk_size
            end = (j+1)*chunk_size
            vals = pests[:,start:end]
            detrend(vals, overwrite_data=True)
            for i in range(63):
                sensor_ind = i+1
                row = vals[i,:]
                detrend(row, overwrite_data=True)
                var = np.var(row)
                if var < var_lim:
                    bracket = Bracket([start, end], sensor_ind, f, f_ind,data_loc)
                    good_bracks.append(bracket)
        print('Pickling brackets for frequency ', f)
        brack_name = get_bracket_name(f, proj_string)
        print('data loc', brack_name)
        with open(brack_name, 'wb') as ff:
            pickle.dump(good_bracks,ff)

def check_bracks():
    with open('/home/fakins/data/swellex/good_brackets_127.pickle', 'rb') as ff:
        bracks = pickle.load(ff)
        fs = set([x.freq for x in bracks])
        fs = list(fs)
        print(fs)
        s1 = [x for x in bracks if x.sensor == 1]
        plt.figure()
        pest = get_pest('/home/fakins/data/swellex/127_pest.npy', 1)
        ind = 0
        for b in s1:
            if ind == 0:
                print(b.data_loc)
                ind += 1
            dom, alpha = b.get_data(pest=pest)
            plt.plot(dom,alpha, color='r')
        plt.savefig('127_check.png')
        
def check_pest(freq, proj_dir='s5'):
    fname = get_pest_fname(freq, proj_dir)
    x = np.load(fname)
    x0 = x[0,:]
    fig = plt.figure()
    plt.plot(detrend(x0))
    plt.savefig('x0.png')
    plt.close(fig)

def check_pest1():
    x = np.load('/home/fakins/data/swellex/127_pest.npy')
    x = detrend(x)
    x0 = x[0,:]
    plt.figure()
    plt.plot(x0)
    plt.savefig('127_pest.png')
    plt.clf()

def collect_brackets(freqs):
    """ I wrote a separate pickle for each
    frequency band...now collect them all into one """
    all_bracks=[]
    for f in freqs:
        with open('/home/fakins/data/swellex/good_brackets_' +str(f) +'.pickle', 'rb') as ff:
            bracks = pickle.load(ff)
        for x in bracks:
            all_bracks.append(x)
    with open('/home/fakins/data/swellex/good_brackets.pickle', 'wb') as ff:
        pickle.dump(ff, all_bracks)
    return
            

if __name__ == '__main__':
            
    freq = sys.argv[1]
    chunk_len = int(sys.argv[2])
    var = float(sys.argv[3])
    proj_string = sys.argv[4]
    freq = int(freq)
    freqs = [freq]
    print('Generating brackets for frequencies ', freqs, ' with chunks of ', chunk_len, ' using a variance limit of ', var)
    #auto_write_indices(cpa=True)
    auto_gen_brackets(freqs, chunk_len, var, proj_string=proj_string)
    check_pest(freq, proj_dir=proj_string)
    #check_pest1()
#    check_bracks()
#    dats = get_pest(49,1)
#    print(dats.size)
#    select_inds(17)

