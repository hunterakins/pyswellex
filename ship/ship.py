import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import mode
from scipy.signal import convolve, firwin, get_window, detrend, find_peaks
from scipy.io import loadmat
from scipy.interpolate import interp1d
from swellex.audio.lse import AlphaEst
import pickle


def load_track(fix_missing=True):
    '''
    Load in the mat file. Not a general function
    Will break if you move s5vla_track
    Returns the range to the vla in the S5 event in km, as well as lat, alo
    '''
    x = loadmat('/home/hunter/data/swellex/ship/s5vla_track.mat')
    x = x['vla_dat'][0][0]
    range_km = x[0]
    lat = x[1]
    lon = x[2]
    time = x[3]
    time = np.array(time, dtype=np.float64)
    if fix_missing == True:
        range_km, lat, lon, time = enter_missing_gps_val(range_km, lat, lon, time)
    return range_km, lat, lon, time

def enter_missing_gps_val(range_km, lat, lon, time):
    """
    no gps_point at 13 minutes into the experiment,
    so just put in a dummy value equal to the average
    of the minute 12 and minute 13 points
    """
    left_side = range_km[12]
    right_side = range_km[13]
    new_val = (left_side +right_side)/2
    range_km = np.insert(range_km, 13, [new_val])

    left_side = lat[12]
    right_side = lat[13]
    new_val = (left_side +right_side)/2
    lat = np.insert(lat, 13, [new_val])

    left_side = lon[12]
    right_side = lon[13]
    new_val = (left_side +right_side)/2
    lon = np.insert(lon, 13, [new_val])

    time = np.insert(time,13, [13])
    return range_km, lat, lon, time
    

def get_velocity(times, range_km):
    dr = range_km[1:] - range_km[:-1]
    dr_meters = dr*1000
    dt = times[1:]-times[:-1]
    dt_seconds = dt*60
    velocities = dr_meters/dt_seconds
    velocities = np.insert(velocities, 0, velocities[0]) # double first value for convenient interpolation
    return velocities

def smooth_velocity(velocity):
    ''' smooth it out for interpolation
    '''
    velocity[1:50] = (velocity[2:51] + velocity[0:49] + velocity[1:50])/3
    velocity[-15:-1] = (velocity[-15:-1] + velocity[-14:])/2
    return velocity

def get_est(pickle_name, c_bias_factor=1):
    with open(pickle_name, 'rb') as f:
        est = pickle.load(f)
    return est

def compare_gps_phase(gps_range_km, ests):
    """ Look at distance bias"""
    plt.figure()
    r0 = (gps_range_km[6] + gps_range_km[7]) / 2
    rf = gps_range_km[-3]
    r_tot = (rf-r0) * 1000
    r_gps = gps_range_km[7:-2]*1000 - r0*1000
    dists = np.zeros(len(ests))
    range_diffs = []
    for i in range(len(ests)):
        thetas = ests[i].thetas
        counter = 0
        print('-----------')
        freq_diffs = []
        for j in range(len(thetas)):
            dist_est = np.sum(thetas[:j])*10
            if (j -3) %6 == 0:
                freq_diffs.append(dist_est)
                counter+=1
        dist_est=  np.sum(thetas[:])*10
        freq_diffs.append(dist_est)# add last mesmt
        plt.plot(freq_diffs[:]-r_gps[:])
        range_diffs.append(freq_diffs)

    bias_factors = []
    for i in range(len(range_diffs)):
        diff = range_diffs[i]
        diffs = np.array(diff)
        H = diffs.reshape(diffs.size)
        norm = H.T@H
        proj = H.T@r_gps
        a = proj/norm
        dom = np.linspace(0, len(diffs), len(diffs))
        bias_factors.append(a) 
    return bias_factors

def compare_gps_phase_deep(gps_range_km, ests):
    """ Look at distance bias"""
    plt.figure()
    r0 = (gps_range_km[6] + gps_range_km[7]) / 2
    rf = gps_range_km[40]
    r_tot = (rf-r0) * 1000
    r_gps = gps_range_km[7:41]*1000 - r0*1000
    dists = np.zeros(len(ests))
    range_diffs = []
    for i in range(len(ests)):
        thetas = ests[i].thetas
        counter = 0
        print('-----------')
        freq_diffs = []
        for j in range(len(thetas)):
            dist_est = np.sum(thetas[:j])*10
            if (j -3) %6 == 0:
                if thetas[j] == 0: # bad msmt
                    freq_diffs.append(0)
                else:
                    freq_diffs.append(dist_est-r_gps[counter])
                counter+=1
        dist_est=  np.sum(thetas[:])*10
        freq_diffs.append(dist_est)# add last mesmt
        plt.plot(freq_diffs[:])
    plt.show()

def proc_shallow(chunk_len, tscc=False, plot_resids=False):
    """ Just a poorly thought out script that 
    a ) applies the heuristic bias of .97
    b ) computes an unweighted average over the different frequency bands
    c ) Throws out some egregious outliers
    d ) looks at the distribution of the residuals
    """

    """ First get the estimates """
    s_freqs = [232, 280,335,385]
    ests = []
    for sf in s_freqs:
        if tscc==False:
            pickle_name = '../audio/pickles/' + str(chunk_len) +'s_shallow_complete_' + str(sf) + '.pickle'
            v_est= get_est(pickle_name)
            v_est.thetas = -np.array(v_est.thetas).reshape(len(v_est.thetas))
            v_est.get_t_grid()
            v_est.add_t0(6.5*60)
            ests.append(v_est)
        else: 
            pickle_name = '../audio/pickles/' + str(chunk_len) + '_' + str(sf) +'.pickle'
            v_est= get_est(pickle_name)
            for i in range(len(v_est.thetas)):
                x = v_est.thetas[i]
                if x == 0:
                    x = np.zeros(v_est.thetas[0].shape)
                    v_est.thetas[i] = x
            v_est.thetas = -np.array(v_est.thetas).reshape(len(v_est.thetas))
            v_est.thetas = v_est.thetas[:-1]
            v_est.get_t_grid()
            ests.append(v_est)


    """Correct the wavenumber bias """
    #for est in ests:
    #    thetas = est.thetas
    #    for i in range(len(thetas)):
    #        if thetas[i] < 0:
    #            thetas[i] = thetas[i]*.97
    #        else:
    #            thetas[i] = thetas[i]*.97


    """ Load GPS for comparison"""
#    plt.plot(v_est.tgrid/60, mean_est)#, yerr=np.sqrt(sigma))

    gps_range_km, lat, lon, gps_time = load_track()
    gps_vel = get_velocity(gps_time, gps_range_km)
        
    plt.figure()
    if tscc == False:
        plt.suptitle("Shallow source estimates")
    else:
        plt.suptitle("Shallow source using full array")

    """ Compute the mean estimate """ 
    ind = False
    mean_est = np.zeros(ests[0].thetas.shape)
    for i in range(len(ests)):
        v_est=ests[i]
        mean_est += v_est.thetas
        plt.plot(v_est.tgrid/60, v_est.thetas, alpha=.4)#, yerr=np.sqrt(sigma))
    mean_est /= len(s_freqs)

    """ Compute the residuals from the mean and identify outliers """
    diffs = np.zeros((len(s_freqs), mean_est.size))

    for i in range(len(s_freqs)):
        v_est= ests[i]
        diffs[i,:] = mean_est- v_est.thetas

    errs = np.var(diffs, axis=0)
    mean_inds = []
    mean_err = np.mean(errs)
    
    """ Identify outliers and update the mean est """
    """ Also update the errorbars """
    outlier_inds = []
    for i in range(len(errs)):
        if errs[i] > 3*mean_err:
            msmts = [x.thetas[i] for x in ests]
            med = np.median(msmts)
            diff = np.array([abs(x-med) for x in msmts])
            bad_ind = np.argmax(diff)
            outlier_inds.append([bad_ind, i])
            okay_msmts = [msmts[i] for i in range(len(ests)) if i != bad_ind]
            outlier_inds.append([bad_ind,i])
            mean_est[i] = np.mean(okay_msmts)
            new_err = np.var([mean_est[i] - x for x in okay_msmts])
            errs[i] = new_err

    """ Plot the velocity estimates with GPS estimates"""
    ind_partition = (40-6.5)*(60//chunk_len)
    ind_partition = int(ind_partition)
    np.save('npys/'+str(chunk_len) + 's_shallow_range_rate.npy', mean_est)
    np.save('npys/'+str(chunk_len) + 's_shallow_range_rate_first.npy', mean_est[:ind_partition])
    np.save('npys/'+str(chunk_len) + 's_shallow_range_rate_cpa.npy', mean_est[ind_partition:])

    plt.plot(v_est.tgrid/60, mean_est, color='black')
    plt.scatter(gps_time[6:], gps_vel[6:],marker='x',color='red',s=9)
    plt.legend([str(sf) + ' Hz est.' for sf in s_freqs[::-1] ] + ['Mean over frequency'] + ['GPS'])
    plt.ylim([3, -3])
    plt.xlabel('Time (mins)')
    plt.ylabel('Range rate (m/s)')
    if tscc == False:
        print("Mean std dev", np.mean(np.sqrt(errs)))
    else:
        print("Mean std dev", np.mean(np.sqrt(errs[78:])))

    plt.errorbar(v_est.tgrid/60, mean_est, np.sqrt(errs),fmt='o',color='black',ecolor='gray',elinewidth=1,ms=3,capsize=2)


    """ Look at residuals from the mean curve to see if they
    look gaussian """
    """ knock out the outliers """
    for j in range(len(errs)):
        for i in range(len(s_freqs)):
            if [i,j] in outlier_inds:
                diffs[i,j] = 0
            

    if plot_resids == True:
        plt.figure()
        ret= plt.hist(diffs.flatten(), 200, density=True)
        vals, x = ret[0], ret[1]
        means = np.mean(diffs, axis=1)
        variances = np.var(diffs, axis=1)
        var = np.var(diffs.flatten())
        mean = np.mean(diffs.flatten())
        vals = np.exp(-np.square(x-mean) / (2*var)) / np.sqrt(2*np.pi*var)
        plt.plot(x, vals)
        plt.xlabel('Residual')
        plt.ylabel('Probability')
    #for i in range(6):
    #    plt.figure()
    #    ret= plt.hist(diffs[i,:], 100, density=True)
    #    vals, x = ret[0], ret[1]
    #    vals = np.exp(-np.square(x-mean) / (2*var)) / np.sqrt(2*np.pi*var)
    #    plt.plot(x, vals)
    #    pass
        
    #plt.show()
    with open('../audio/pickles/shallow_ests_unbiased.pickle', 'wb') as f:
        pickle.dump(ests,f)
    return mean_est

def proc_deep(chunk_len, tscc=False, plot_resids=False):
    """ A copy of proc_shallow for the deep source more or less
    Basically trying to make a nice plot comparing GPS and 
    phase derived range rate
    """

    """ First get the estimates """
    d_freqs = [235, 283,338,388]
    ests = []
    flag = False
    for df in d_freqs:
        if tscc==False:
            pickle_name = '../audio/pickles/' + str(chunk_len) + 's_deep_complete_' + str(df) + '.pickle'
            v_est= get_est(pickle_name)
            v_est.thetas = -np.array(v_est.thetas).reshape(len(v_est.thetas))
            v_est.get_t_grid()
            """ deal with gaps in data """
            if flag == False:
                inds = [i for i in range(len(v_est.thetas)) if v_est.thetas[i] == 0]
                flag = True
            v_est.add_t0(6.5*60)
            v_est.thetas = [v_est.thetas[i] for i in range(len(v_est.thetas)) if i not in inds]
            v_est.thetas = np.array(v_est.thetas).reshape(len(v_est.thetas))
            v_est.tgrid = np.array([v_est.tgrid[i] for i in range(len(v_est.tgrid)) if i not in inds])
            ests.append(v_est)
        else: 
            pickle_name = '../audio/pickles/' + str(chunk_len) + '_' + str(df) +'.pickle'
            v_est= get_est(pickle_name)
            for i in range(len(v_est.thetas)):
                x = v_est.thetas[i]
                if x == 0:
                    x = np.zeros(v_est.thetas[0].shape)
                    v_est.thetas[i] = x
            v_est.thetas = -np.array(v_est.thetas).reshape(len(v_est.thetas))
            v_est.thetas = v_est.thetas[:-1]
            v_est.get_t_grid()
            ests.append(v_est)


    """ Biases are assumed to by around .97 """

    gps_range_km, lat, lon, gps_time = load_track()
    gps_vel = get_velocity(gps_time, gps_range_km)
    #ests = load_shallow_unbiased()
#    bias_factors = compare_gps_phase(gps_range_km,  ests)
    bias_factors=[.97]*len(d_freqs)
    for i in range(len(ests)):
        ests[i].thetas = bias_factors[i]*ests[i].thetas

    plt.figure()
    if tscc == False:
        plt.suptitle("Deep source estimates")
    else:
        plt.suptitle("Deep source all sensors")

    """ Compute the mean estimate and also plot the phase derived velocities"""
    ind = False
    mean_est = np.zeros(ests[0].thetas.shape)
    for i in range(len(ests)):
        v_est=ests[i]
        thetas = v_est.thetas
        mean_est += thetas
        plt.plot(v_est.tgrid/60, v_est.thetas, alpha=.4)#, yerr=np.sqrt(sigma))



    mean_est /= len(d_freqs)

    """ remove outlier """
    np.save('npys/'+str(chunk_len) + 's_deep_range_rate.npy', mean_est)
    ind_partition = (40-6.5)*(60//chunk_len)
    ind_partition = int(ind_partition)
    np.save('npys/'+str(chunk_len) + 's_deep_range_rate.npy', mean_est)
    np.save('npys/'+str(chunk_len) + 's_deep_range_rate_first.npy', mean_est[:ind_partition])
    np.save('npys/'+str(chunk_len) + 's_deep_range_rate_cpa.npy', mean_est[ind_partition:])

    """ Plot the velocity estimates with GPS estimates"""

    plt.scatter(gps_time[6:], gps_vel[6:],marker='x',color='red',s=9)
    plt.legend([str(df) + ' Hz est.' for df in d_freqs[::-1] ] + ['Mean over frequency'] + ['GPS'])
    plt.ylim([3, -3])
    plt.xlabel('Time (mins)')
    plt.ylabel('Range rate (m/s)')

    """ Compute the residuals from the mean """
    diffs = np.zeros((len(d_freqs), mean_est.size))

    for i in range(len(d_freqs)):
        v_est= ests[i]
        diffs[i,:] = mean_est- v_est.thetas

    errs = np.var(diffs, axis=0)
    mean_inds = []
    mean_err = np.mean(errs)
    
    """ Identify outliers and update the mean est """
    """ Also update the errorbars """
    outlier_inds = []
    for i in range(len(errs)):
        if errs[i] > 3*mean_err:
            msmts = [x.thetas[i] for x in ests]
            med = np.median(msmts)
            diff = np.array([abs(x-med) for x in msmts])
            bad_ind = np.argmax(diff)
            outlier_inds.append([bad_ind, i])
            okay_msmts = [msmts[i] for i in range(len(ests)) if i != bad_ind]
            outlier_inds.append([bad_ind,i])
            mean_est[i] = np.mean(okay_msmts)
            new_err = np.var([mean_est[i] - x for x in okay_msmts])
            errs[i] = new_err

    
    errs = np.var(diffs, axis=0)
    plt.plot(v_est.tgrid/60, mean_est, color='black')
    plt.errorbar(v_est.tgrid/60, mean_est, np.sqrt(errs),fmt='o',color='black',ecolor='gray',elinewidth=1,ms=3,capsize=2)


    """ Look at residuals from the mean curve to see if they
    look gaussian """
    if plot_resids == True:
        plt.figure()
        ret= plt.hist(diffs.flatten(), 200, density=True)
        vals, x = ret[0], ret[1]
        means = np.mean(diffs, axis=1)
        variances = np.var(diffs, axis=1)
        var = np.var(diffs.flatten())
        mean = np.mean(diffs.flatten())
        vals = np.exp(-np.square(x-mean) / (2*var)) / np.sqrt(2*np.pi*var)
        plt.plot(x, vals)
        plt.xlabel('Residual')
        plt.ylabel('Probability')
    #for i in range(6):
    #    plt.figure()
    #    ret= plt.hist(diffs[i,:], 100, density=True)
    #    vals, x = ret[0], ret[1]
    #    vals = np.exp(-np.square(x-mean) / (2*var)) / np.sqrt(2*np.pi*var)
    #    plt.plot(x, vals)
    #    pass
    
    plt.figure()
    plt.plot(v_est.tgrid[::12][6:]/60, mean_est[::12][6:], color='black')
    plt.scatter(gps_time[6:][:-3], gps_vel[6:][:-3],marker='x',color='red',s=9)
    print(gps_vel[6:][:-3].shape, mean_est[::12][6:].shape)
    x = gps_vel[6:-3]
    y = mean_est[::12][6:]
    plt.figure()
    plt.plot(y/x)
        
    with open('../audio/pickles/deep_ests_unbiased.pickle', 'wb') as f:
        pickle.dump(ests,f)
    return mean_est

def form_shallow_vel(chunk_len, pickle_root='/oasis/tscc/scratch/fakins/'):
    """ Modification of proc_shallow
    Goal is to look up the results of  
    the least squares for each of the high
    shallow frequencies (see below)
    Then combine them into one average
    velocity, throwing out outliers,
    and saving the result on the server 
    Input 
    chunk_len - integer
        the length of the chunks used in the 
        least squares
    pickle_root - string
            root location of estimate outputs
    Output 
    saves a numpy array with the velocity estimates
    """

    """ First get the estimates """
    s_freqs = [232, 280,335,385]
    ests = []
    for sf in s_freqs:
            pickle_name =pickle_root + str(chunk_len) + '_' + str(sf) +'.pickle'
            v_est= get_est(pickle_name)
            for i in range(len(v_est.thetas)):
                x = v_est.thetas[i]
                if x == 0:
                    x = np.zeros(v_est.thetas[0].shape)
                    v_est.thetas[i] = x
            v_est.thetas = -np.array(v_est.thetas).reshape(len(v_est.thetas))
            v_est.thetas = v_est.thetas[:-1]
            v_est.get_t_grid()
            ests.append(v_est)

    """ Compute the mean estimate """ 
    ind = False
    mean_est = np.zeros(ests[0].thetas.shape)
    for i in range(len(ests)):
        v_est=ests[i]
        mean_est += v_est.thetas
    mean_est /= len(s_freqs)

    """ Compute the residuals from the mean and identify outliers """
    diffs = np.zeros((len(s_freqs), mean_est.size))

    for i in range(len(s_freqs)):
        v_est= ests[i]
        diffs[i,:] = mean_est- v_est.thetas

    errs = np.var(diffs, axis=0)
    mean_inds = []
    mean_err = np.mean(errs)
    
    """ Identify outliers and update the mean est """
    """ Also update the errorbars """
    outlier_inds = []
    for i in range(len(errs)):
        if errs[i] > 3*mean_err:
            msmts = [x.thetas[i] for x in ests]
            med = np.median(msmts)
            diff = np.array([abs(x-med) for x in msmts])
            bad_ind = np.argmax(diff)
            outlier_inds.append([bad_ind, i])
            okay_msmts = [msmts[i] for i in range(len(ests)) if i != bad_ind]
            outlier_inds.append([bad_ind,i])
            mean_est[i] = np.mean(okay_msmts)
            new_err = np.var([mean_est[i] - x for x in okay_msmts])
            errs[i] = new_err

    np.save(pickle_root+str(chunk_len) + 'sw_rr.npy', mean_est)
    np.save(pickle_root+str(chunk_len) + 'sw_rr_errs.npy', errs)

    return mean_est

def load_shallow_unbiased(tscc=False):
    if tscc==False:
        with open('../audio/pickles/shallow_ests_unbiased.pickle', 'rb') as f:
            ests= pickle.load(f)
    else:
        ests = []
        for f in [232, 280,  335, 385]:
            with open('../audio/pickles/5_' + str(f) + '.pickle', 'rb') as f:
                est = pickle.load(f)
                ests.append(est)
    return ests

def shallow_ar():
    """
    Look at acf of random motion about polynomial velocity model
    Get variance of signal around mean
    """
    ests = load_shallow_unbiased()
    mean_acf = 0
    mean_est, _ = get_mean_est(ests)
    plt.figure()
    for x in ests:
        speeds = x.thetas
        speeds -= np.mean(mean_est)
        window = get_window('hann', speeds.size)
        speeds *= window
        fvals = np.fft.rfft(speeds)
        freqs = np.fft.rfftfreq(speeds.size, 10) # ten second spacing
        acf = np.square(abs(fvals))
        plt.plot(freqs, acf)
        plt.show()
        if type(mean_acf) == type(0):
            mean_acf = acf
        else:
            mean_acf += acf
    mean_acf /= len(ests)
    plt.figure()
    plt.plot(freqs,mean_acf)
    plt.show()

def get_mean_est(ests):
    for i in range(len(ests)):
        if i == 0:
            mean_est = np.zeros(ests[i].thetas.size)
        mean_est += ests[i].thetas
    mean_est /= len(ests)
    diffs = np.zeros((len(ests), mean_est.size))
    for i in range(len(ests)):
        diffs[i,:] = mean_est - ests[i].thetas
    var = np.var(diffs, axis=0)
    return mean_est, var

def get_white_noise_var(ests):
    """
    Assuming a simple constant velocity model
    driven by uncorrelated white noise, 
    get estimate of white noise variance
    """
    v_vars = []
    for x in ests:
        vels = x.thetas
        var = np.var(vels)
        v_vars.append(var)
    return sum(v_vars) / len(v_vars)

def simplest_kalman(gps_var):
    """    
    Combine GPS measurements with phase range rate measurements
    in a simple Kalman filter model with uncorrelated white noise
    driving variation in the velocity 
    I model GPS range msmt as corrupted by white gaussian noise
    of variance gps_var supplied by user
    
    The phase rate msmt variance is estimated from assuming 
    independence of the frequency bands processed
    """

    """ First load msmt stuff """

    """Get rdot estimates """
    rdot_msmts = load_shallow_unbiased()

    """ Get range rate measurement mean and variances """
    mean, msmt_var = get_mean_est(rdot_msmts)
    rdot = mean
    dom = np.linspace(6.5, 40, mean.size)

    """ Get GPS position measurements"""
    range_km, lat, lon, time = load_track()
    range_m = range_km[7:41]*1e3
    time = time[7:41]

    """ Form  the list of msmts """
    z_list = [np.array([[x],[0]]) for x in mean]
    for i in range(len(z_list)):
        if ((i - 3) % 6) == 0:
            z_list[i][1,0] = range_m[int((i-3)//6)]

    """ Form the list of msmt covariance matrices """
    C_list = [np.array([[msmt_var[i], 0], [0, gps_var]]) for i in range(len(z_list))]


    """ Form state noise covariance (constant in time) """
    S = np.square(2.4)
    wgn_var = get_white_noise_var(rdot_msmts)
    alpha = np.sqrt((S-wgn_var)/S)
    delta = 10 # 10 second time step
    Q = np.array([[wgn_var, delta/2*wgn_var],[delta/2*wgn_var, delta*delta/4*wgn_var]])

    """ Form A matrix that upates state """
    A = np.array([[alpha, 0], [10, 1]])


    """ Get initial state_estimate corresponding to minute 6.5"""
    r0 = (range_km[6]+range_km[7])*1e3 / 2 
    rdot0 = rdot[0]
    skk = np.array([[rdot0],[r0]])

    """ Get initial covariance error """
    Ck = C_list[0]
    Mkk = Ck

    num_iters = mean.size
    sk_log = [skk]
    Mk_log = [Mkk]
    for i in range(1,num_iters):
        """ Initial prediction step """
        skk1 = A@skk
        Mkk1 = A@Mkk@A.T + Q


        """ Form Hk """
        if (i - 3)% 6 == 0: 
            H = np.identity(2)
            """ Else just have info about rdot"""
        else:
            H = np.zeros((2,2))
            H[0,0] = 1
        """ Kalman Gain """
        Kn = Mkk1@H.T@np.linalg.inv(C_list[i] + H@Mkk1@H.T)

        """ Fetch zk"""
        zk = z_list[i]
        skk = skk1 + Kn@(zk - H@skk1)
        Mkk = (np.identity(skk.size) - Kn@H)@Mkk1
        Mk_log.append(Mkk)
        sk_log.append(skk)
    return sk_log

def assemble_ests(chunk_len, source_depth):   
    """ I run my phase estimation on two sections of the data separately
    This scripts merges them into one object """
    """
    Input - 
        chunk_len - int or float
            the chunk size in the phase estimation thing    
        source_depth - string
            'shallow' or 'deep'
    Output -
        None
        saves a pickled copy of the AlphaEst object
        with the combined estimates
    """

    if source_depth == 'shallow':
        freqs = [232, 280,335,385]
    if source_depth == 'deep':
        freqs = [235, 283,338,388]

    if chunk_len == 10:
        """ I have 4 pickled estimates I want to combine into
        1 nice object
        """
        ests = []
        for f in freqs:
            """ Get 6.5 minute piece """
            pickle_name = '../audio/pickles/chunk_10s_auto_' + source_depth + '_'+str(f)+'_freqs.pickle'
            v_est= get_est(pickle_name)
            t1 = v_est.thetas
            s1 = v_est.sigmas
            """ Get last estimate """
            pickle_name = '../audio/pickles/chunk_10s_auto_' + source_depth + '_'+str(f)+'_freqs_last_msmt.pickle'
            v_est= get_est(pickle_name)
            t2 = v_est.thetas
            s2 = v_est.sigmas

            """ Get 40 minut """
            pickle_name = '../audio/pickles/chunk_10s_auto_' + source_depth + '_'+str(f)+'_freqs_cpa.pickle'
            v_est= get_est(pickle_name)
            t3 = v_est.thetas
            s3 = v_est.sigmas

            if source_depth == 'shallow':
                """ Get last msmt """
                pickle_name = '../audio/pickles/chunk_10s_auto_' + source_depth + '_'+str(f)+'_freqs_cpa_last_msmt.pickle'
                v_est= get_est(pickle_name)
                t4 = v_est.thetas
                s4 = v_est.sigmas
            
                all_thetas = t1 + t2 + t3 + t4
                all_sigmas = s1 + s2 + s3 + s4
                complete_est = AlphaEst(v_est.chunk_len, all_thetas, all_sigmas)
            else: 
                all_thetas = t1 + t2 + t3 
                all_sigmas = s1 + s2 + s3 
                complete_est = AlphaEst(v_est.chunk_len, all_thetas, all_sigmas)

            with open('../audio/pickles/10s_' + source_depth + '_complete_' + str(f) + '.pickle', 'wb') as f: 
                pickle.dump(complete_est, f)

    if chunk_len == 5:
        """ There are two pickle estimates to combine """
        ests = []
        for f in freqs:
            """ Get 6.5 minute piece """
            pickle_name = '../audio/pickles/chunk_5s_auto_' + source_depth + '_'+str(f)+'.pickle'
            v_est= get_est(pickle_name)
            t1 = v_est.thetas
            s1 = v_est.sigmas

            """ Get 40 minut """
            pickle_name = '../audio/pickles/chunk_5s_auto_' + source_depth + '_'+str(f)+'_cpa.pickle'
            v_est= get_est(pickle_name)
            t2 = v_est.thetas
            s2 = v_est.sigmas

            all_thetas = t1 + t2
            all_sigmas = s1 + s2
            complete_est = AlphaEst(v_est.chunk_len, all_thetas, all_sigmas)
            with open('../audio/pickles/5s_' + source_depth + '_complete_' + str(f) + '.pickle', 'wb') as f: 
                pickle.dump(complete_est, f)

def simple_comp_test1():
    est = get_est('../audio/pickles/5_385.pickle')
    est.get_t_grid()
    est_comp = get_est('../audio/pickles/5s_shallow_complete_385.pickle')
    t_grid_comp = est_comp.get_t_grid()
    est_comp.add_t0(6.5*60)
    est_comp.thetas = np.array(est_comp.thetas)
    est_comp.thetas = est_comp.thetas.reshape(est_comp.thetas.size)
    plt.plot(est.tgrid, est.thetas)
    plt.plot(est_comp.tgrid, est_comp.thetas)
    plt.show()


def plot_script():
    """ Script for playing around and looking at plots""" 
    #mean_full = proc_shallow(2, tscc=True)
    #np.save('npys/sw_2.npy', mean_full)
    #mean_full = proc_shallow(5, tscc=True)
    #np.save('npys/sw_5.npy', mean_full)
    #mean = proc_shallow(5)
    mean = proc_deep(5, tscc=True)
    #mean = proc_deep(2, tscc=True)
    #mean_full = proc_deep(5)
    plt.show()

    #x = np.load('npys/1024sw_rr.npy')
    #plt.figure()
    #plt.plot(x[:-6])


    #plt.figure()
    #fs = 1500/1024
    #ind = int(45*60 * fs)
    #first_chunk = x[5:ind]
    #first_chunk = detrend(first_chunk)
    #fx = np.fft.rfft(first_chunk)
    #freqs = np.fft.rfftfreq(len(first_chunk), 1024/1500)
    #plt.plot(freqs, np.square(abs(fx)))
    
    plt.show()

if __name__ == '__main__':
    plot_script()

    #chunk_len = int(sys.argv[1])
    #est = form_shallow_vel(chunk_len)
    #plot_script()
   # plt.figure()
    #plt.plot(est)
    #plt.savefig('/oasis/tscc/scratch/fakins/pic.png')
    #plt.clf()
    #gps_var = 1
    #sk_log = simplest_kalman(.1)
    #rdot_ests = [x[0] for x in sk_log]
    #r_ests = [x[1] for x in sk_log]

    #rdot_msmts = load_shallow_unbiased()
    #mean, msmt_var = get_mean_est(rdot_msmts)
    #plt.plot(rdot_ests)
    #plt.plot(mean)
    #plt.show()

    #plt.plot(r_ests)
    #plt.show()


