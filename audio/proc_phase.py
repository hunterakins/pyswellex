import numpy as np
from matplotlib import pyplot as plt
import pickle
from scipy.signal import detrend, convolve, firwin, hilbert
import matplotlib._color_data as mcd

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

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
        
def get_pest(f0, s_ind):
    """
    Fetch the pickled phase estimate corresponding to 
    frequency f0 and sensor s_ind
    """
    fname = 'pickles/kay_s' + str(s_ind) + 'pest_' + str(int(f0)) + '.pickle'
    with open(fname, 'rb') as f:
        p_est = pickle.load(f)
    return p_est

def get_alt_pest(f0, s_ind):
    """
    Fetch the pickled phase estimate corresponding to 
    frequency f0 and sensor s_ind
    """
    fname = 'pickles/kay_s' + str(s_ind) + 'alt_pest_' + str(int(f0)) + '.pickle'
    with open(fname, 'rb') as f:
        p_est = pickle.load(f)
    return p_est

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

def form_good_pest(f, brackets,sensor_ind=1,detrend_flag=False):
    """
    Take in the intervals specified in the list of 
    brackets and return the portions of the time domain
    and the portions of the phase estimates that 
    correspond  
    Input
    f - float
        A frequency that you've processed
    brackets - list of list
        List of the left and right minute for each of
        the low noise segments of that frequency
    Output - 
    dom_segs -list of np arrays
        each element is the relevant domain for
        corresponding phase estimates
    pest_segs - list of np arrays
        each element is a np array
        of phase ests
    """
    pest =get_pest(f, sensor_ind)
    if detrend_flag == True:
        pest = detrend(pest)
    dom = np.linspace(6.5, 40, pest.size)
    inds = []
    size = 0
    dom_segs = []
    pest_segs = []
    for bracket in brackets:
        dom = np.linspace(6.5, 40, pest.size)
        dt = dom[1]-dom[0]
        if bracket[0] > bracket[1]:
            print(f, bracket)
            raise(ValueError, 'bracket isn\'t increasing')
        li = int((bracket[0]-6.5) / dt)
        ri = int((bracket[1]-6.5) / dt)
        seg = pest[li:ri]
        dom_seg = dom[li:ri]
        pest_segs.append(seg)
        dom_segs.append(dom_seg)
    num_samps = sum([x.size for x in dom_segs])
    return dom_segs, pest_segs

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
    x,y = form_good_pest(f, brackets,sensor_ind,detrend_flag=True)
    for dom_seg, pest_seg in zip(x,y):
        line = (plt.plot(dom_seg, pest_seg, label=str(f)))
    plt.xlabel('Time (min)')
    plt.ylabel('Unwrapped phase (radians)')
    plt.show()


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

if __name__ == '__main__':
    freqs= [49, 64, 79, 94, 109, 112,127, 130, 148, 166, 201, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
#    freqs= [148, 166, 201, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 

    sensor = 9
#    for f in freqs:
#        print('-------------- ' + str(f) + ' hz')
#        fig, ax = plt.subplots()
#        pest =get_pest(f, sensor)
#        dom = np.linspace(6.5, 40, pest.size)
#        fig.suptitle(str(f))
##       # compare_to_snr(f, 3)
#        pts = ax.scatter(dom[::100], detrend(pest[::100]), s=1)
#        selector = SelectFromCollection(ax, pts)
#        plt.show()
#        selector.disconnect()
#        for x in selector.inds:
#            print('[' + str(dom[100*x[0]]) + ', ' + str(dom[100*x[1]]) + '],')

    from indices.sensor9 import mins            
#    for i, f in enumerate(freqs):
#        print(f)
#        brackets = mins[i]
#        print(brackets)
#        check_filter(f, brackets, sensor_ind=9)


    plot_lin_reg_good(freqs[:14],mins[:14],9)
    plt.show()
