import numpy as np
#from matplotlib import pyplot as plt
from scipy.signal import hilbert, firwin,convolve, detrend
from scipy.ndimage import convolve1d
import sys
#import pickle 

'''
Description:
Implement the Djuric and Kay phase unwrapping algorithm
in  Parameter Estimation of Chirp Signals

Date: 4/8/2020

Author: Hunter Akins
'''


def kay_preproc(s_n, fc, order, df):
    """
    Input 
    s_n - np array
        each row is a sensor time series
    fc - float
        center frequency of narrowband filter
    order - int
        filter length
    df - float
        1/2 the bandwidth of the narrowband filter
    Output
    u_n - numpy array with same dims as s_n
        hilbert transform of narrowband filter output
    """
    filt = firwin(order, [fc-df, fc+df], fs=1, pass_zero=False, window='hann')
    u_n = hilbert(s_n)
    u_n = convolve(u_n, filt, mode='same')
    return u_n

def kay_preproc_vec(s_n, fc, order, df,save=False, froot=None):
    """
    Input 
    s_n - np array
        each row is a sensor time series
    fc - float
        center frequency of narrowband filter
    order - int
        filter length
    df - float
        1/2 the bandwidth of the narrowband filter
    Output
    u_n - numpy array with same dims as s_n
        hilbert transform of narrowband filter output
    if save is True, the bandpassed filtered data is saved
    at froot
    """
    filt = firwin(order, [fc-df, fc+df], fs=1, pass_zero=False, window='hann')
    s_n = convolve1d(s_n, filt, mode='constant')
    if save == True:
        print('Saving the band pass filtered time series for ', str(1500*fc))
        np.save(froot + str(int(1500*fc)) + '_ts.npy', s_n)
    u_n = hilbert(s_n)
    return u_n

def diff_seq(u_n):
    y_n = u_n[1:]*(u_n[:-1].conj())
    z_n = y_n[1:]*(y_n[:-1].conj())
    return y_n, z_n

def diff_seq_vec(u_n):
    y_n = u_n[:,1:]*(u_n[:,:-1].conj())
    z_n = y_n[:,1:]*(y_n[:,:-1].conj())
    return y_n, z_n

def p_est(z_n):
    p = np.angle(z_n)
    return p

def undiff(p0, p, y_n):
    """
    Integrate phases back
    Input -
    p0 - float
    p - np array 
        phase of the compelx sequence (radians)
        represents the double differenced phase
    y_n - np array of complex 128
        conjugated offset seq from diff_seq
    Output
     - np array
        integrated phase   
    """
    """Get single differenced phase estimates """
    dp = np.angle(y_n)
    """ Get sequence phi2 -phi0, phi3-phi1, ..."""
    single_diff = p + dp[:-1]
    """ Get total number of phases and init phase array """
    N = y_n.size + 1
    p_final = np.zeros(N)
    p_final[0] = p0
    p_final[1] = dp[0] + p0
    for i in range(2, N):
        p_final[i] = p_final[i-1]+single_diff[i-2]
    return p_final

def undiff_vec(p0, p, y_n):
    """
    Integrate phases back
    Input -
    p0 - float
    p - np array 
        phase of the compelx sequence (radians)
        represents the double differenced phase
    y_n - np array of complex 128
        conjugated offset seq from diff_seq
    Output
     - np array
        integrated phase   
    """
    """Get single differenced phase estimates """
    dp = np.angle(y_n)
    """ Get sequence phi2 -phi0, phi3-phi1, ..."""
    single_diff = p + dp[:,:-1]
    """ Get total number of phases and init phase array """
    N = y_n.shape[1] + 1
    p_final = np.zeros((y_n.shape[0],N))
    p_final[:,0] = p0
    p_final[:,1] = dp[:,0] + p0
    for i in range(2, N):
        p_final[:,i] = p_final[:,i-1]+single_diff[:,i-2]
    return p_final

def get_p_est(sensor, fn, dfn, order):
        """ Estimate the phase """
        un = kay_preproc(sensor, fn, order, dfn)
        yn, zn = diff_seq(un)
        ddp = np.angle(zn) 
        p0 = np.angle(un[0])
        p_est = undiff(p0, ddp, yn)
        return p_est

def get_vec_p_est(x, fn, dfn, order, save=False, froot=None):
    """ Estimate phase at freq fn
    but vectorized 
    Input 
        x - numpy ndarray
            each row is a sensor msmst sequence
        fn - float
            center freq
        dfn - float
            bandwidht
        order - int
            filter order
    Output 
        p_est - ndarray
            phase estimates, same size as x"""
    un = kay_preproc_vec(x, fn, order, dfn, save, froot)
    yn, zn = diff_seq_vec(un)
    ddp = np.angle(zn) 
    p0 = np.angle(un[:,0])
    p_est = undiff_vec(p0, ddp, yn)
    return p_est

def test_vec_p_est():
    """
    Test the vector p_est function
    for the supercomputer """
    s1 = np.load('npy_files/sensor' + str(1) + 'cpa.npy') # zero mean
    s2 = np.load('npy_files/sensor' + str(2) + 'cpa.npy') # zero mean
    """
    Only use first 3 minutes
    """
    s1 = s1[:3*60*1500]
    s2 = s2[:3*60*1500]
    s1 = s1.reshape(s1.size)
    s2 = s2.reshape(s2.size)
    x = np.zeros((2, s1.size))
    x[0,:] = s1
    x[1,:] = s2
    f0 = 49
    df = .5
    dt =  1/1500
    fn = f0*dt 
    dfn = df*dt
    pest = get_vec_p_est(x, fn, dfn, 1024)
    print(pest.shape)
    plt.plot(detrend(pest[0,:]))
    plt.plot(detrend(pest[1,:]))

    pest1 = get_p_est(s1, fn, dfn, 1024)
    plt.plot(detrend(pest1), color='r')
    plt.show()

def data_run(freqs, sensor_inds, df):
    """
    Estimate phase of received data 
    Input 
    freqs - list
    Frequencies of bands of interests
    sensor_inds - list
    indices of sensors of interest (index at 1)
    df - float
    bandwidth/2 of the filter applied to the time domain data
    """
    fs = 1500
    dt =  1/fs
    """ Load in the data """
    for sensor_ind in sensor_inds:
        sensor = np.load('npy_files/sensor' + str(sensor_ind) + 'cpa.npy') # zero mean
        sensor -= np.mean(sensor)
        """ Normalize the data (makes it easier to plot) """
        sensor = sensor/np.sqrt(np.var(sensor))
        sensor = sensor.reshape(sensor.size)
        N = sensor.size

        """ For each band """
        for f0 in freqs:
            """ Convert to dimensionless """
            fn = f0*dt 
            dfn = df*dt
            """ Choose order of FIR filter
            You want it short enough to resolve the wavenumbers
            """
            order =  1024 
            """ Estimate the phase """
            p_est = get_p_est(sensor, fn, dfn, order)
            fname = '/media/hunter/ExtHard/pickles/kay_s' + str(sensor_ind) + 'cpapest_' + str(int(f0)) + '.pickle'
            print(fname)
            with open(fname, 'wb') as f:
                pickle.dump(p_est, f)

def lfm_chirp():
    """
    Make a 200 millisecond lfm sweep
    From 2.5 kHz to 4.5 kHz
    alpha = 2000/.2 = 10000
    t = .2  
    """
    fs = 40000
    T = .2
    dt = 1/fs
    f0 = 2.5*1e3
    alpha = 2000/T
    lfm = lambda n: np.cos((f0 + alpha*dt*n)*dt*n)
    N = fs*T
    dom = np.linspace(0,N,N)
    vals = lfm(dom)
    ptrue = (f0+alpha*dt*dom)*dt*dom
    un = hilbert(vals)
    yn, zn = diff_seq(un)
    ddp = np.angle(zn) 
    p0 = np.angle(un[0])
    p_est = undiff(p0, ddp, yn)
    plt.plot(p_est)
    plt.plot(ptrue)
    plt.show()

def sim_est():
    t_series = np.load('npy_files/49_time_field.npy')
    sensor_1 = t_series[0,:]
    sensor_1 = sensor_1/np.sqrt(np.var(sensor_1))
    sensor_1 += np.random.randn(sensor_1.size)*.3
    un = sensor_1
    yn, zn = diff_seq(un)
    ddp = np.angle(zn) 
    p0 = np.angle(un[0])
    p_est = undiff(p0, ddp, yn)
    plt.subplot(211)
    plt.plot(p_est)
    plt.subplot(212)
    plt.plot(sensor_1)
    plt.show()
    return p_est

def tscc_pest(freqs, save=False, froot=None):
    """
    For each frequency in freqs, 
    take in the swellex numpy array 
    and generate an array of the phases ests for 
    that freq

    Input 
        freqs - list of floats/ints    
            frequencies at which to perform the 
            phase estimation
    Output 
        None
        saves an array for each frequency in freqs
        to /home/fakins/data with the name
        freq_pest.npy 
        if save is True, the bandpass filtered
        data is saved at froot """ 
    x = np.load('/oasis/tscc/scratch/fakins/data/swellex/s5_good.npy')
    dt = 1/1500
    df = .5
    for f in freqs:
        fn = f*dt 
        dfn = df*dt
        order =  1024 
        """ Estimate the phase """
        pest = get_vec_p_est(x, fn, dfn, order, save, froot)
        print('Saving phase estimates for ', f, ' Hz')
        np.save('/oasis/tscc/scratch/fakins/data/swellex/' + str(f) +'_pest.npy', pest)
    return

def tscc_coh_sum(freqs):
    """
    For each frequency in freqs, 
    take in the swellex numpy array s5.npy
    sum coherently over array
    then perform phase est.
    """
    x = np.load('/home/fakins/data/s5.npy')
    """ reduce to a single row"""
    x = np.sum(x, axis=0)
    print(x.shape, 'shape of reduced array')
    dt = 1/1500
    df = .5
    for f in freqs:
        fn = f*dt 
        dfn = df*dt
        order =  1024 
        """ Estimate the phase """
        pest = get_p_est(x, fn, dfn, order)
        np.save('/oasis/tscc/scratch/fakins/data/' + str(f) +'_coh_pest.npy', pest)
        print('First pest samps', pest[:10])
        

    
    

if __name__ == '__main__':
    #sim_est()
    #sensor_inds = [1,2,3,4,5,6,7,9,9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    #sensor_inds = [13,14,15,16]
#    sensor_inds = [17, 18, 19, 20, 21]
#    freqs = [49, 64, 79]#, 94, 112, 130]#, 130, 75, 82]
    #df = .5
    #freqs = [49, 64, 79, 94, 109, 112, 127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    #freqs = [127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    #data_run(freqs, sensor_inds, df)
    freq = sys.argv[1]
    freqs = [int(freq)]
    print('running tscc pest at ', freqs)
    tscc_pest(freqs, save=True, froot='/oasis/tscc/scratch/fakins/')
#    tscc_coh_sum(freqs)


#    lfm_chirp()
