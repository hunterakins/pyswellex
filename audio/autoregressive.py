import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert, detrend, find_peaks
from scipy.linalg import solve_toeplitz, toeplitz
import scipy.linalg as la
import sys
import time 
from swellex.audio.ts_comp import get_nb_fname

'''
Description:
Routines for implementing an autoregressive estimator for the PSD of the data
on each sensor in the array. 

The golden nugget here is run_esprit and the 
esprit implementation

Date: 
07/22/2020

Author: Hunter Akins
'''


def get_chunk(data, min0, min1):
    """ Get the chunk from min0 to min1 """
    ind0, ind1 = int(min0*60*1500), int(min1*60*1500)
    if data.shape[0] == data.size:
        data = data.reshape(1,data.size)
    data = data[:,ind0:ind1]
    return data

def fourier_psd(data):
    freqs, ft = np.fft.fftfreq(data.size, 1/1500), np.fft.fft(data)
    psd = np.square(abs(ft))
    return freqs, psd

def retain_relevant(freq, freqs, psd):
    """
    Only keep the bins nearby freq 
    """
    freq_inds = [i for i in range(len(freqs)) if abs(freqs[i] - freq) < .5]
    return freqs[freq_inds], psd[freq_inds]

def est_autocorr(data, model_order, start_lag=0):
    """
    Perform simple autocorrelation estimation for lag up
    to model order m
    Input
    data - np array, possibly 1d
    model_order - int
        number of lags
    start_lag - int
        helpful for recursive implementations, if you've laread
        computed lags up to L, start_lag=L+1 will only work onwards
    """
    if data.shape[0] == data.size: 
        data = data.reshape(1, data.size)
    m = model_order
    rxx = np.zeros(m, dtype=np.complex128)
    N = data.shape[1]
    """ FOr each lag """
    for m in range(start_lag, model_order):
        sum_vals= 0
        for n in range(N-m):
            sum_vals += np.sum(data[:,n+m]*(data[:,n].conj()))
        rxx[m] = sum_vals/(N-m)
    return rxx

def update_autocorr(data, rxx, new_order):
    """
    Given rxx, which contains estimates of the lag up to
    l = rxx.size-1, expand it to contain estimates to include
    lags up to new_order-1
    """
    start_lag = rxx.size
    if new_order < start_lag:
        raise ValueError('New lag is less than old lkag')
    new_rxx = est_autocorr(data, new_order, start_lag = start_lag)
    full_rxx = np.zeros(new_order,dtype=np.complex128)
    full_rxx[:start_lag] = rxx
    full_rxx[start_lag:] = new_rxx
    return full_rxx

def make_data_mat(data, model_order):
    num_rows = data.size - model_order
    num_cols = model_order
    mat = np.zeros((num_rows, num_cols), dtype=np.complex128)
    h = np.zeros(num_rows, dtype=np.complex128)
    for i in range(num_rows):
        mat[i,:] = -data[i:i+model_order:][::-1]
        h[i] = data[i+model_order]
    mat = np.matrix(mat)
    return mat, h
    
def solve_yw(data, model_order):
    """
    Solve yule-walker equations for data
    Input
    data- array like
        ...
    model_order - int
        number of AR coeffs
    output
    an - array
    var - float
        estimated noise variance
    """
    mat,h = make_data_mat(data, model_order)
    print(mat.shape)
    inv = np.linalg.inv(mat.H@mat)@mat.H
    an = inv@h
    an = an.T
    #an = solve_toeplitz(rxx[:-1], rxx[1:])
    #var = rxx[0] + np.sum(an*(rxx[1:].conj()))
    var =1
    return an, var

def get_psd(f_grid, delta_t, an, var):
    """
    Get power spectral density from estimated
    autoregressive parameters
    """
    numer  =delta_t*abs(var)
    psd = np.zeros(f_grid.size)
    t = delta_t * np.linspace(1, len(an), len(an))
    an =an.reshape(1, an.size)
    for i, f in enumerate(f_grid):
        fvals = np.exp(-complex(0,1)*2*np.pi*f*t)
        fvals.reshape(fvals.size, 1)
        denom = 1 + (an@fvals)[0]
        psd[i] = numer/np.square(abs(denom))
    psd = psd / np.max(psd)
    return psd

def get_naive_psd(f_grid, data, delta_t = 1/1500):
    psd = np.zeros(f_grid.size)
    t = delta_t * np.linspace(0, data.size-1, data.size)
    print(t[1]-t[0], delta_t)
    for i, f in enumerate(f_grid):
        comb = np.exp(-complex(0,1)*2*np.pi*f*t)
        psd[i] = np.square((abs(comb@data)))
    psd = psd / np.max(psd)
    return psd

def pseudo_spec(v, df=.1, fs=1500):
    """
    Input
    v - numpy matrix
        columns are the noise eigenvectors
    df - float
        frequency spacing (fs = 1)
    Output 
    psd - numpy array
    """
    T = 1/df
    N = T*fs
    N = np.power(2, int(np.log2(N))+1)
    fft_vals = np.fft.fft(v, n=N, axis=0)
    freqs = np.fft.fftfreq(N, 1/1500)
    power = np.square(abs(fft_vals))
    power_sum = np.sum(power, axis=1)
    psd = 1/power_sum
    return freqs, psd
    
def music(data, num_freqs, M, df=.1, fs=1500):
    """
    Compute the pseudospectrum for the data
    looking for num_freqs using M eigenvectors
    Input -
    data - np 1d array
        time series with harmonic content
    num_freqs - int
        number of harmonic elements
    M - int
        lag order to compute
    df - float
        pseudospectrum spacing
    fs - float
        sampling rate (Hz)
    """
    rxx = est_autocorr(data, M)
    mat = toeplitz(rxx)
    lam, v = la.eigh(mat)
    #plt.figure()
    #plt.scatter(np.linspace(0, lam.size-1, lam.size), lam)
    #plt.ylim(0, np.max(lam)*1.1)
    snr_proxy = lam[-1]
    noise_dim = M - num_freqs
    noise_v = v[:,:noise_dim]
    freqs, psd = pseudo_spec(noise_v, df, fs)
    df = freqs[1]-freqs[0]
    return freqs, psd, snr_proxy
   
def esprit_make_mat(data, M):
    """
    Make the data matrix for ESPRIT 
    if data is multidimensional, stack the data
    Rows are expected to be sensors
    Input 
        data -numpy 1d or nd array
        M - int
    """
    num_rows = data.size-M
    if len(data.shape) > 1:
        data = data.reshape(data.size)
    X = np.zeros((num_rows, M), dtype=data.dtype) 
    for i in range(M):
        X[:,i] = data[i:i+num_rows]
    return X
    
def esprit(data, p, M,timer=False):
    """
    Given the data, implement Esprit
    Looking for p sinusoids using data vectors of length M
    Input - 
        data - numpy ndarray
            potentially a higher dimensional array,
            in which case there are two ways to approach it
        p - int
            number of complex exponentials hiding in the data
        M - length of the data records
    Output -
    """
    dt = 1/1500
    if timer == True:
        now = time.time()
    X = esprit_make_mat(data,M)
    if timer == True:
        print('making data mat esp', time.time()-now)
        now = time.time()
    U, s, VH = la.svd(X,full_matrices=False)
    if timer == True:
        print('svd 1', time.time() - now)
        now = time.time()
    VH = np.matrix(VH)
    VHs = VH[:p,:]
    VH1 = VHs[:, 1:]
    VH2 = VHs[:, :-1]
    """ DO some TLS"""
    X_tls = np.zeros((M-1, 2*p), dtype=VH1[0,0].dtype)
    X_tls[:,:p] = (VH1.H)
    X_tls[:,p:] = (VH2.H)
    if timer == True:
        print('tls stuff ', time.time() - now)
        now = time.time()
    U,s,VH_tls = la.svd(X_tls)
    if timer == True:
        print('svd 2', time.time()-now)
        now = time.time()
    U_tild = np.matrix(VH_tls).H
    U12_tild = U_tild[:p,p:]
    U22_tild = U_tild[p:,p:]
    Psi_tls = -U12_tild@la.inv(U22_tild)
    Psi = 1/(VH1@(VH1.H))*VH1@(VH2.H)
    eigs = la.eigvals(Psi)
    f = np.angle(eigs)/2/np.pi/dt
    eigs = la.eigvals(Psi_tls)
    f = np.angle(eigs)/2/np.pi/dt
    if timer == True:
        print('final calcs', time.time()-now)
    return f

def check_ts(chunk,freq):
    """
    The data is a chunk of the time series
    Determine if the chunk is good by looking for nulls
    To do so, look at variation of the amplitude, estimated
    for every ten cycles
    """
    chunk_len = chunk.size
    dt = 1/1500
    T = 20/freq
    N = int(T/dt)
    num_subs = chunk_len // N
    masks = np.zeros((num_subs, chunk_len))
    for i in range(num_subs):
        masks[i, N*i:N*i+N] = 1
    amps = masks@abs(chunk)
    amp_range = np.max(amps)/np.min(amps)
    return amp_range

def get_top(dats, freq, num):
    """
    Use check_ts as template
    to measure amplitude variation of all
    the 63 chunks
    Then keep the ...you guessed it...TOP TEN WINNERSSS!!!
    """
    chunk_len = dats.shape[1]
    dt = 1/1500
    T = 20/freq
    N = int(T/dt)
    num_subs = chunk_len // N
    masks = np.zeros((num_subs, chunk_len))
    for i in range(num_subs):
        masks[i, N*i:N*i+N] = 1
    amps = masks@abs(dats.T)
    max_amp, min_amp = np.max(amps, axis=0), np.min(amps, axis=0)
    diffs = max_amp - min_amp
    top_ten = np.argsort(diffs)[:num]
    return dats[top_ten,:]
    
def test_music(freq, p, model_order, T):
    varrs = []
    lfreq = freq-1
    rfreq = freq+1
    for t_start in np.arange(0, 10, 1):
        fs = 1500
        data = np.load('npy_files/'+str(freq) + '_short.npy') #this is data from 35 to 40 minutes into S5
        snrs = []
        x1 = get_chunk(data, t_start/60, t_start/60+T/60)
        x1 = hilbert(x1)
        x1 = x1[0,:]
        x1 = x1.reshape(1, x1.size)

        #t = np.linspace(t_start, t_start + T- 1/fs, T*fs)
        #print(t[1]-t[0])
        #dats = np.cos(2*np.pi*127.2*t)
        #x2, y= fourier_psd(dats)
        #dats = hilbert(dats)
        #x2 = dats
        print('N = ', x1.shape[1])
        freqs, psd, snr_proxy = music(x1, p, model_order, df=.001)
        snrs.append(np.var(x1))
        llim, rlim = np.argmin([abs(lfreq-x) for x in freqs]), np.argmin([abs(rfreq - x) for x in freqs])
        freqs = freqs[llim:rlim]
        psd = psd[llim:rlim]
        psd /= np.max(psd)
        #plt.plot(freqs, psd)

        inds = find_peaks(psd)
        peak = freqs[inds[0]]
        print(peak)
        if len(peak) != 0:
            plt.scatter(t_start, peak)
        #plt.show()
    plt.suptitle('T='+str(T)+', f=' + str(freq) + ' music')

def test_esprit(freq, p, model_order, T):
    varrs = []
    lfreq = freq-1
    rfreq = freq+1
    for t_start in np.arange(0, 5, 1):
        guesses=[]
        for i in range(63):
            fs = 1500
            data = np.load('npy_files/'+str(freq) + '_short.npy') #this is data from 35 to 40 minutes into S5
            snrs = []
            x1 = get_chunk(data, t_start/60, t_start/60+T/60)
            x1 = hilbert(x1)
            x1 = x1[i,:]
            #x1 = x1[:10,:]
            #plt.plot(x1.real)
            #plt.show()

            t = np.linspace(t_start, t_start + T- 1/fs, T*fs)
            dats = np.cos(2*np.pi*freq*t)
            x2, y= fourier_psd(dats)
            dats = hilbert(dats)
            x2 = dats
            f= esprit(x1, p, model_order)
            plt.scatter(t_start, (f-freq)/freq*1500,color='b')
            guesses.append(f)
        guesses=np.array(guesses)
        plt.scatter(t_start, (np.median(guesses)-freq)/freq*1500,color='r')


    plt.suptitle('T='+str(T)+', f=' + str(freq) + ' Esprit')

def run_esprit(freq, p, model_order, N, delta_n, alt=False):
    """
    Run ESPRIT on the tscc supercomputer for frequency freq
    on sliding windows of length N and spcing delta_n
    Input 
        freq - int
            source frequency under consideration
        p - int
            number of freqs to estimate
        model_order - int
            number of elements used in the data matrix formation
        N - int
            num_samps of windows
        delta_t - float
            spacing of windows
    """
    now = time.time()
    x = np.load(get_nb_fname(freq, alt=alt))
    print(get_nb_fname(freq, alt=alt))
    #x = np.load('npy_files/'+str(freq) + '_short.npy') #this is data from 35 to 40 minutes into S5
    #x = np.load('npy_files/'+str(freq) + '_short_hilb.npy') #this is data from 35 to 40 minutes into S5
    #x = x[:, :15000]
    x = hilbert(x)
    now = time.time()
   
    num_to_keep = 5
    num_samps = x.shape[1]
    num_ests = (num_samps- N)//delta_n
    f_hat = np.zeros((num_to_keep, num_ests))
    err = np.zeros((num_to_keep, num_ests))
    amp = np.zeros((num_to_keep, num_ests))
    dt = 1/1500
    t = np.arange(0, N*dt, dt)
    print('Running esprit on ', freq, ' band. Window size is ', N, ' spaced by ', delta_n)
    min_amp_f = []
    min_err_f = []
    print('initializing variables')
    print(time.time() - now)
    for i in range(num_ests):
        lind, rind = i*delta_n, i*delta_n + N
        dats = np.copy(x[:,lind:rind])
        dats /= np.std(dats, axis=1).reshape(63,1)
        dats =get_top(dats, freq, 5)
        for j in range(num_to_keep):
            tmp = dats[j,:]
            #tmp /= np.std(tmp) # std is sqrt(A) / 2 for pure sine wave
            peak = np.max(tmp[:100])
            p0 = np.angle(tmp[0]/peak)
            f= esprit(tmp, p, model_order)
            dat_hat = np.sqrt(2)*np.exp(complex(0,1)*2*np.pi*f*t)
            err_var = np.sqrt(np.sum(abs(dat_hat - tmp)))
            amp_var = check_ts(tmp,freq)
            f_hat[j,i] = f
            err[j,i] = err_var
            amp[j,i] = amp_var
        np.save('/oasis/tscc/scratch/fakins/fests/'+str(freq) + '_' + str(N) + '_' + str(delta_n) + '_fhat.npy', f_hat)
        np.save('/oasis/tscc/scratch/fakins/fests/'+str(freq) + '_' + str(N) + '_' + str(delta_n) + '_fhat_err.npy', np.array(err))
        np.save('/oasis/tscc/scratch/fakins/fests/'+str(freq) + '_' + str(N) + '_' + str(delta_n) + '_fhat_amp.npy', np.array(amp))
            
        #fig, axes = plt.subplots(3,1)
        #axes[0].plot(f_hat[:,i])
        #minaf = f_hat[np.argmin(amp[:,i]),i]
        #min_amp_f.append(minaf)
        #minerrf = f_hat[np.argmin(err[:,i]),i]
        #min_err_f.append(minerrf)
        #axes[0].scatter(np.argmin(amp[:,i]), minaf, color='r')
        #axes[0].scatter(np.argmin(err[:,i]), minerrf, color='g')
        #axes[1].plot(err[:,i])
        #axes[2].plot(amp[:,i])
        #plt.show()
    #min_amp_f = np.array(min_amp_f)
    #min_err_f = np.array(min_err_f)
    #print(min_amp_f - min_err_f)
    #plt.plot(min_amp_f, color='r')
    #plt.plot(min_err_f,color='g')
    #plt.show()
        
        np.save('/oasis/tscc/scratch/fakins/fests/'+str(freq) + '_' + str(N) + '_' + str(delta_n) + '_fhat.npy', f_hat)
        np.save('/oasis/tscc/scratch/fakins/fests/'+str(freq) + '_' + str(N) + '_' + str(delta_n) + '_fhat_err.npy', np.array(err))

def rm_outliers(fests, num_std):
    """
    Trim outliers from the set
    Return the median of the trimmed series
    Input 
        fests - np array
            each row is a vector of msmts froma  sensor
        num_std - float
            number of standard deviations to clip
    Output
        medians - np array
    """
    var = np.var(fests, axis=0)
    means = np.mean(fests, axis=0)
    diffs= abs(fests-means)
    medians = np.zeros(var.size)
    for i in range(fests.shape[1]):
        inds = np.where(diffs[:,i] < num_std*np.sqrt(var[i]))
        medians[i] = np.median(fests[:,i][inds])
        #medians[i] = np.median())
    return medians

def load_fest(freq, N=3000, delta_n=1500,tscc=False):
    """ 
    Load the frequency estimate for freq for the given interval and spacing N and delta_n
    """
    if tscc == True:
        root = '/oasis/tscc/scratch/fakins/fests/'+str(freq) + '_' + str(N) + '_' + str(delta_n) + '_fhat' 
    else:
        root = 'npy_files/fests/'+str(freq) + '_' + str(N) + '_' + str(delta_n) + '_fhat' 
    fhat = np.load(root+'.npy')
    err = np.load(root+'_err.npy')
    amp = np.load(root+'_amp.npy')
    return fhat, err, amp

def check_fest(freq, N, delta_n):
    """ Compare to walker?  """
    #i1 = int(15*60*1500 / delta_n)
    #i2 = int(40*60*1500/delta_n)
    i1, i2 = 0, -1
    x,y,z = load_fest(freq, N, delta_n)
    x = x[:,i1:i2]
    y = y[:,i1:i2]
    z = z[:,i1:i2]
    best_inds_err = np.argmin(y, axis=0)
    best_inds_amp = np.argmin(z, axis=0)
    i = np.linspace(0, best_inds_err.size-1, best_inds_err.size,dtype=int)
    fig, axis = plt.subplots(1,1)
    axes = [axis]
    t = i*delta_n/1500 / 60
    axes[0].plot(t, x[best_inds_err,i],color='b')
    #axes[0].plot(t, x[best_inds_amp,i],color='g')
    fig.suptitle(str(freq))
    #fig1, axes = plt.subplots(2,1)
    #axes[0].acorr(x[best_inds_amp, i], detrend=detrend)
    #axes[1].acorr(x[best_inds_err, i], detrend=detrend)
    #fig1.suptitle(str(freq))
    return fig, axes, t
    
def comp_fest(freqs, N, delta_n):
    """ Compare to walker?  """
    i1 = int(15*60*1500 / delta_n)
    i2 = int(40*60*1500/delta_n)
    #i1, i2 = 0, -1
    fig, axes = plt.subplots(1,1)
    fig1, axes1 = plt.subplots(1,1)
    check = 0
    for freq in freqs:
        x,y,z = load_fest(freq, N, delta_n)
        x = x[:,i1:i2]
        y = y[:,i1:i2]
        z = z[:,i1:i2]
        best_inds_err = np.argmin(y, axis=0)
        best_inds_amp = np.argmin(z, axis=0)
        i = np.linspace(0, best_inds_err.size-1, best_inds_err.size,dtype=int)
        if check == 0:
            ref_x = x[best_inds_err,i]
            ratio = 1
            check = 1
        else:
            curr_x = x[best_inds_err, i]
            ratio = curr_x@ref_x.T / (np.square(np.linalg.norm(curr_x)))
            print(ratio)
        axes.plot(i, (x[best_inds_err,i]*ratio))
        #axes[1].plot(i, z[best_inds_amp, i], color='b')
        #axes[2].plot(i, y[best_inds_amp, i], color='g')
        axes1.acorr(x[best_inds_amp, i], detrend=detrend)

def comp_two_fest(freqs, N, delta_n):
    """ Compare to walker?  """
    i1 = int(15*60*1500 / delta_n)
    i2 = int(40*60*1500/delta_n)
    #i1, i2 = 0, -1
    fig, axes = plt.subplots(1,1)
    fig1, axes1 = plt.subplots(1,1)
    check = 0
    x,y,z = load_fest(freqs[0], N, delta_n)
    x = x[:,i1:i2]
    y = y[:,i1:i2]
    z = z[:,i1:i2]
    best_inds_err = np.argmin(y, axis=0)
    i = np.linspace(0, best_inds_err.size-1, best_inds_err.size,dtype=int)
    ref_x = x[best_inds_err,i]
    ratio = 1
    check = 1

    
    x,y,z = load_fest(freqs[1], N, delta_n)
    x = x[:,i1:i2]
    y = y[:,i1:i2]
    z = z[:,i1:i2]
    curr_x = x[best_inds_err, i]
    ratio = curr_x@ref_x.T / (np.square(np.linalg.norm(curr_x)))
    axes.plot(i, (curr_x*ratio))
    axes.plot(i, ref_x)
    #axes[1].plot(i, z[best_inds_amp, i], color='b')
    #axes[2].plot(i, y[best_inds_amp, i], color='g')
    #axes1.acorr(x[best_inds_amp, i], detrend=detrend)
    axes1.plot(i, ref_x - curr_x*ratio)

def comp_filter():
    """
    I ran ESPRIT on 385 with .2 Hz bandwidth
    and .1 
    Compare the results 
    """
    freq = 385
    i1 = int(6.5*60*1500/750)
    i2 = int(60*60*1500/750)
    narr_x, y, z = load_fest(freq, N=1500, delta_n=750)
    best_inds_err = np.argmin(y, axis=0)
    i = np.linspace(0, best_inds_err.size-1, best_inds_err.size, dtype=int)
    narr_x = narr_x[best_inds_err,i]
    root = 'npy_files/fests/385_1500_750_fhat' 
    x = np.load(root+'_orig.npy')
    y = np.load(root+'_err_orig.npy')
    z = np.load(root+'_amp_orig.npy')
    best_inds_err = np.argmin(y, axis=0)
    x = x[best_inds_err,i]
    plt.plot(x[i1:i2])
    plt.plot(narr_x[i1:i2])
    plt.show()

if __name__ == '__main__':
    freq = int(sys.argv[1])
    N = int(sys.argv[2])
    delta_n = int(sys.argv[3])
    T = 1
    p = 1 
    M = 40

    run_esprit(freq, p, M, N, delta_n, alt=False)

#plt.figure()
#test_esprit(freq,p,M,T)
#freq = 335
#plt.figure()
#test_esprit(freq,p,M,T)
#plt.show()
