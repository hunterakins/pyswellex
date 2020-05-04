import os
import numpy as np
from matplotlib import pyplot as plt
from indices.sensors import freqs, mins
from proc_phase import form_good_pest, form_good_alpha, get_pest
from scipy.signal import detrend
import pickle
import time


'''
Description:
Perform the least square estimation of the ship velocity

Date: 
4/26


Author: Hunter Akins
'''

def form_noise_est(brackets):
    """
    Get sample variance of detrended "good segments"
    For list of these sample variances to use as estimates of the noise 
    variance for each sensor/frequency
    Input -
    brackets- list of list of list of list
        Each sensor has a list of lists corresponding to each freq
        Each freq has a list of 2-element lists
        The first element is the start minute of the good seg,
        second element is the end minute
    Output -
        noise_ests - list of list of lists
        noise_ests[0][0][0] is the variance around a line of the
        first element at 49 Hz for the first good segment of data
    """
    noise_ests = []
    for i, sensor_bracket in enumerate(brackets):
        freq_ests = []
        for j, freq in enumerate(freqs):
            brackets = sensor_bracket[j]
            dom_segs, pest_segs = form_good_pest(freq, brackets, i+1,detrend_flag=True)
            curr_noise_ests = []
            for seg in pest_segs:
                seg = detrend(seg)
                noise_var = np.var(seg)
                curr_noise_ests.append(noise_var)
            freq_ests.append(curr_noise_ests)
        noise_ests.append(freq_ests)
    return noise_ests

def form_alpha_noise_est(brackets,freqs,use_pickle=False,c=1500):
    """
    Get sample variance of detrended "good segments" in the alpha
    variable
    For list of these sample variances to use as estimates of the noise 
    variance for each sensor/frequency
    Input -
    brackets- list of list of list of list
        Each sensor has a list of lists corresponding to each freq
        Each freq has a list of 2-element lists
        The first element is the start minute of the good seg,
        second element is the end minute
    Output -
        noise_ests - list of list of lists
        noise_ests[0][0][0] is the variance around a line of the
        first element at 49 Hz for the first good segment of data
    """
    if use_pickle==True:
        with open('noise/alpha_noise.pickle','rb') as f:
            est = pickle.load(f)
            return est
    noise_ests = []
    for i, sensor_bracket in enumerate(brackets):
        freq_ests = []
        for j, freq in enumerate(freqs):
            brackets = sensor_bracket[j]
            dom_segs, pest_segs = form_good_alpha(freq, brackets, i+1,c)
            curr_noise_ests = []
            for k in range(len(pest_segs)):
                dom, seg = dom_segs[k], pest_segs[k]
                if dom.size == 0:
                    print(brackets)
                seg = detrend(seg)
                noise_var = np.var(seg)
                curr_noise_ests.append(noise_var)
            freq_ests.append(curr_noise_ests)
        noise_ests.append(freq_ests)
    with open('noise/alpha_noise.pickle', 'wb') as f:
        pickle.dump(noise_ests, f)
    return noise_ests

def get_Kn(Sigma, h, sigma_sample):
    """
    Get the sequential least squares gain matrix
    K from Sigma, the current error covariance matrix,
    h, the newest model row as a column matrix!, and sigm_sample,
     the estimated noise on the newest measurement
    Return 
    Kn the gain matrix
    """ 
    Sh = Sigma@h
    numer = Sh
    denom = sigma_sample + h.T@Sh
    Kn = numer/denom
    return Kn

def update_theta(theta_c, K, alpha, h):
    """
    Get newest estimate theta (as column vec)
    given gain matrix K, new measurement alpha,
    and new model row h.T
    """
    theta_new = theta_c + K*(alpha - h.T@theta_c)
    return theta_new

def update_Sigma(Sigma_curr,K,h):
    """
    Update error covariance matrix Sigma from
    the current error cov Sigma_curr,
    the gain matrix K, and the new model row
    h.T
    """
    n = h.size
    Sigma_new = (np.identity(n) - K@h.T)@Sigma_curr
    return Sigma_new
    
def get_h(curr_sec, num_f, df, poly_order):
    """
    Form the new model row h.T at time t=curr_sec with num-f freq comp
    Input -
        curr_sec - float
        num_f - int
            number of frequencies to include model
    Output -
    h - numpy array (column)
    """
    h = np.ones(((2*num_f+1)*poly_order,1))
    for i in range(poly_order):
        h[i,0] = np.power(curr_sec,i+1)
    """ Populate the columns with the sines and cosines """
    if num_f > 0:
        for j in range(poly_order):
            multiplier = h[i,0]
            for i in range(1, num_f+1):
                vals = multiplier*np.cos(2*np.pi*i*df*curr_sec)
                h[poly_order + j*2*num_f + 2*i-2,0] = vals
                vals = multiplier*np.sin(2*np.pi*i*df*curr_sec)
                h[poly_order + j*2*num_f + 2*i-1,0] = vals
    return h

class Bracket:
    """ Bracket of good data """
    def __init__(self, bracket, sensor_ind, freq, f_ind, alpha_var):
        """ 
        Input 
            bracket - list, bracket[0] is start in mins, bracket[1] is end
            in mins
            sensor_ind - int >= 1
            freq - int 
                frequency band
            f_ind - int
                index of the frequency in the ''canonical list''
        """
        """ I had to correct for an error I made in assigning 
        the minute values to the brackets....
        I used a domain from 6.5 to 40, but I should've used
        a domain from 6.5 to 40 - (1/1500 / 60) which
        is the sampling rate in mins
        I correc the mins, then use them to get the right index
        """
        dm = 1/1500/60
        self.start_min = 6.5 + (bracket[0]-6.5)/1.0000003316612551
        self.end_min = 6.5 + (bracket[1]-6.5)/1.0000003316612551
        """ index of start and end of bracket"""
        self.start = int((self.start_min-6.5) / dm)
        self.end = int((self.end_min-6.5) / dm)
        self.bracket = [self.start, self.end]
        self.sensor = sensor_ind
        self.freq = freq
        self.f_ind = f_ind
        self.alpha_var = alpha_var
        self.alpha0=None
        self.dom = None
        self.data = None

    def add_alpha0(self, alpha0):
        """
        add an alpha0
        """
        self.alpha0 = alpha0

    def get_data(self,c=1500):
        dom_segs, alpha_segs = form_good_alpha(self.freq, [self.bracket], self.sensor+1, c)
        if type(self.alpha0) != type(None):
            x0= alpha_segs[0][0]
            diff = self.alpha0 - x0
            alpha_segs[0] += diff
        self.data = alpha_segs[0]
        self.dom = dom_segs[0]
        return dom_segs[0], alpha_segs[0]

    def __repr__(self):
        return "Bracket for sensor {:d} at frequency {:d} starting at minute {:.2f} and ending at minute {:.2f}".format(self.sensor + 1, self.freq, self.start_min, self.end_min)

def get_row_sizes(opening_bracks, batch_size):
    max_ind = batch_size
    num_rows = 0 # to count total num
    row_sizes = [] # to hold the size for each bracket
    for x in opening_bracks:
        dom, alpha = x.get_data()
        """ If the bracket is fully contained
        in the batch chunk, the whole data segment
        will be used """
        if x.end < batch_size:
            num_samps = x.end-x.start
        else:
            num_samps = max_ind - x.start 
        num_rows+=num_samps
        row_sizes.append(num_samps)
    return num_rows, row_sizes
  
def form_first_H(opening_bracks, num_f, df, poly_order, batch_size):
    """
    Form the model for the first batch LS estimate
    Input -
        opening_bracks - list of Brackets
            that open in the opening period
        num_f - int
            number of frequencies to include model
    Output -
    H - numpy matrix
        matrix consisting of big model for all the brackets
        truncated to the smallest bracket
    """

    """ First figure out how big H will be"""
    max_ind = batch_size
    num_rows, row_sizes = get_row_sizes(opening_bracks, batch_size)
  
    """ Iterate through the brackets
    and populate an H model matrix """ 
    dt = 1/1500
    t = np.linspace(0, dt*batch_size, batch_size)
    """ H that contains all possible rows needed """
    H = np.ones((batch_size, (2*num_f+1)*poly_order))
    for i in range(poly_order):
        H[:,i] = np.power(t, i+1)
    """ Populate the columns with the sines and cosines """
    if num_f > 0:
        for j in range(poly_order):
            multiplier = H[:,j]
            for i in range(1, num_f+1):
                vals = multiplier*np.cos(2*np.pi*i*df*t)
                H[:,poly_order + 2*j*num_f +2*i-2] = vals
                vals = multiplier*np.sin(2*np.pi*i*df*t)
                H[:,poly_order + 2*j*num_f + 2*i-1] = vals
    """ Make the big H by selecting the appropriate 
    row from H"""
    big_H = np.ones((num_rows, (2*num_f+1)*poly_order))
    curr_row = 0 # keep track of where to add new values
    for i in range(len(opening_bracks)):
        curr_brack = opening_bracks[i]
        start_ind, end_ind = curr_brack.start, curr_brack.end
        if end_ind > max_ind:
            end_ind = max_ind
        row_size = row_sizes[i]
        relevant_H = H[start_ind:end_ind,:]
        big_H[curr_row:curr_row+row_size] = relevant_H
        curr_row += row_size
    return big_H

def form_first_alpha(opening_bracks, batch_size):
    """
    Get a big column vector of the first
    measurements of alpha
    """
    max_ind = batch_size
    """ First determine size of column vec and 
    contribution of each bracket """
    num_rows, row_sizes = get_row_sizes(opening_bracks, batch_size)

    alphas = np.zeros((num_rows,1))
    curr_row = 0
    for i in range(len(opening_bracks)):
        x = opening_bracks[i]
        row_size = row_sizes[i]
        y = alphas[curr_row:curr_row+row_size, 0]
        alphas[curr_row:curr_row+row_size,0] = x.data[:row_size]
        curr_row += row_size
    return alphas
    
def get_H_inv(H, opening_bracks, batch_size):
    """
    Use covariance of brackets and initial thing of
    H to compute H pseudo inverse
    Hpsuedo is inv(H.T@Cinv@H)@H.T@Cinv
    C is diagonal with highly structured blocksj
    """
    """first compute block size """
    num_rows, row_sizes = get_row_sizes(opening_bracks, batch_size)
    """
    Compute Cinv H and H.T @ Cinv """
    CiH = np.zeros(H.shape)
    CiH[:,:] = H
    curr_row = 0
    for i in range(len(opening_bracks)):
        brack = opening_bracks[i]
        var = brack.alpha_var
        row_size = row_sizes[i]
        CiH[curr_row:curr_row+row_size,:] *= 1/var 
        curr_row += row_size
    inv = np.linalg.inv(H.T@CiH)
    pinv = inv @ CiH.T
    cov = inv
    return pinv, cov
        
def get_brackets(mins, n_ests, freqs, use_pickle=True):
    if use_pickle == True:
        with open('brackets/brackets.pickle', 'rb') as f:
            brackets = pickle.load(f)
            return brackets
    else:
        b_objs = []
        for i in range(len(mins)):
            n_est = n_ests[i]
            for j in range(len(freqs)):
                freq_ests = mins[i][j]
                for k,bracket in enumerate(freq_ests):
                    b_obj = Bracket(bracket, i, freqs[j], j, n_est[j][k])
                    b_objs.append(b_obj)
        with open('brackets/brackets.pickle', 'wb') as f:
            pickle.dump(b_objs, f) 
        return b_objs

def get_initial_est(b_list, num_f, df, poly_order, batch_size, start_ind=0):
    """
    get estimate of theta and Sigma to seed
    the sequential least squares algorithm
    use the first batch_size*dm minutes of data to accomplish this

    Input 
    b_list - list of Brackets
    num_f - int
        num frequencies
    df - float
        freq spacing
        whether to reverse the phase samples...
        in beta 

    Output
    theta - numpy array (column)
        model parameter estimates
    cov - numpy matrix
        param cov matrix
    
    """

    """ Figure out which brackets contain data in this interval """
    opening_bracks = [x for x in b_list if (0 <= (x.start - start_ind)) and ((x.start - start_ind) <= batch_size)]

    """ Form a model matrix from these brackets """
    H = form_first_H(opening_bracks, num_f, df, poly_order, batch_size,start_ind)
    """ COmbine the msmts from these brackets into one ''supervector'' """
    alphas = form_first_alpha(opening_bracks, batch_size)

    Hinv, cov = get_H_inv(H, opening_bracks, batch_size)
    theta0 = Hinv@alphas
    return theta0, cov, opening_bracks
    
def sort_by_end(b_list):
    """
    Take in a list of brackets
    Sort them in order of their closing minute
    """
    inds = np.argsort(np.array([x.bracket[1] for x in b_list]))
    sorted_list = [b_list[x] for x in inds]
    return sorted_list

def sort_by_start(b_list):
    """
    Take in a list of brackets
    Sort them in order of their opening minute
    """
    inds = np.argsort(np.array([x.bracket[0] for x in b_list]))
    sorted_list = [b_list[x] for x in inds]
    return sorted_list

def seq_least_squares(b_list, num_f, df, poly_order, theta, Sigma, start_ind, batch_size=30000, open_bracks=[],model_sec_start=0):
    """
    Do sequential least squares
    Input -
        b_list - list of Brackets 
            data to use
        num_f - int
            number of frequencies in the model
        df - float
            frequency spacing
        poly_order - int
            order of polynomial to fit
        theta - numpy array (column)
            initial parameter estimates
        Sigma - numpy matrix
            seed initial error covariance
        start_ind - int
            reference to which part of the data to use
    """
   
    """ Get total number of samples in the data record """ 
    _ = get_pest(49, 1)
    num_samples =  _.size

    """ Form the domain in seconds, since I use seconds in my
        model (frequencies in Hz) """
    ds = 1/1500
    sec_dom = np.linspace(0, (40-6.5)*60-ds, num_samples)

    start_sec = sec_dom[start_ind]
    print('Starting sequential least_squares at ', start_sec/60)

    """    Sort the brackets by end_min  """
    sorted_b_last = sort_by_end(b_list)

    """ Get the open brackets """
    curr_ind = start_ind
    curr_sec = start_sec
    
    """
    Remove the brackets that end before the start min """
    while sorted_b_last[0].end <= start_ind:
        sorted_b_last.pop(0)


    """ Generate a sorted list by first min """
    sorted_b_first = sort_by_start(sorted_b_last)

    """ Truncate brackets at the start ind. Those
        that contain the start ind are popped off the
        sorted_b_first list and added to open_brackets """


    while sorted_b_first[0].start< start_ind:
        x = sorted_b_first.pop(0)
        """
        Truncate at start_ind for all  brackets """ 
        x.bracket[0] = start_ind
        """ init data objects """
        _,_ = x.get_data()
        open_bracks.append(x)

    """
    Sort open brackets by their closing time
    """
    open_bracks = sort_by_end(open_bracks)

    """ Make sure they are all clipped to start_ind """
    for x in open_bracks:
        """ Clip them to start_ind """
        trunc_length = start_ind - x.start
        x.dom = x.dom[trunc_length:]
        x.data = x.data[trunc_length:]


    """
    Loop through whole system and update theta
    """
    final_ind = curr_ind + batch_size
    alpha_list = []
    """ Model sec as a different variable allows me to
    test out doing chunks at a later time...for now 
    I don't distinguish """
#    model_sec = sec_dom[curr_ind] commented out since i'm trying the chunk
    model_sec = model_sec_start
    while curr_ind <= final_ind:
        tmp = time.time()        

        """ Remove brackets that closed """
        while open_bracks[0].end <= curr_ind:
            x = open_bracks.pop(0)
            """ Double check that there is no 
            data left in that bracket """
            #print(len(x.data))
            #print('removed bracket', x)

        """ add brackets that have opened """
        while curr_ind >= sorted_b_first[0].start:
            """ Pop first element off sorted_b_first """
            x= sorted_b_first.pop(0)
            
            """ Offset the data to agree with the rest of
            this estimation """
            """ I'm commenting this out for the purpose of
            comparing batch and sequential as a test"""
            x.add_alpha0(curr_alpha)

            """ initialize data attributes """
            _,_ = x.get_data() 
            open_bracks.append(x)
            """ Make sure open bracks are sorted
                by closing time """
            open_bracks = sort_by_end(open_bracks)

        """ Create model row h.T for the current time """
        h = get_h(model_sec, num_f, df, poly_order)
        for brack in open_bracks:
            alpha = brack.data[0]
            """ pop off top element of data """
            brack.data = np.delete(brack.data, 0)
            brack.dom = np.delete(brack.dom, 0)
            """ Compute gain matrix """
            K = get_Kn(Sigma, h, brack.alpha_var)
            """ Compute model parameter estimates and 
            estimate covariance """
            theta = update_theta(theta, K, alpha, h)
            Sigma = update_Sigma(Sigma, K, h)
        curr_alpha = (h.T@theta)[0,0]
        #alpha_list.append(curr_alpha)
        curr_ind += 1
        curr_sec += ds
        model_sec += ds

    """
    Sorted_b_last has popped off all the leftovers, so it contains
    brackets that are either currently open or haven't been opened yet
    I remove the elements of sorted_b_last that are in open_bracks
    """
    leftover_b_list = [x for x in sorted_b_last if x not in open_bracks]
    """ Open bracks have had their domains and values modified, 
    so update their start index to reflect that """
    for x in open_bracks:
        x.start = curr_ind
    return theta, Sigma, alpha_list, curr_ind, open_bracks, leftover_b_list
            
def form_alpha_est(times, num_f, df, theta):
    """
    Tkae in a set of times, number of frequencies, f spacing df, 
    and parameter estimate theta
    run the forward model ont he times
    Input -
        opening_bracks - list of Brackets
        num_f - int
            number of frequencies to include model
    Output -
    alpha_est the forward model
    """
    """find the size of the smallest bracket
        and the corresponding domain
        while we loop through, grab all the samples I will 
        need for the inverse """
    """ Form mini H"""
    t = times
    H = np.ones((t.size, 2*num_f+1))
    H[:,0] = t
    """ Populate the columns with the sines and cosines """
    if num_f > 0:
        for i in range(1, num_f+1):
            vals = t*np.cos(2*np.pi*i*df*t)
            H[:,2*i-1] = vals
            vals = t*np.sin(2*np.pi*i*df*t)
            H[:,2*i] = vals
    return H@theta

def compare_sls_to_batch(): 
    """ Form minute domain, secon domain """
    pest = get_pest(49, 1)
    dm = 1/1500/60
    min_dom = np.linspace(6.5, 40-dm, pest.size)
    sec_dom = np.linspace(0, (40-6.5-dm)*60, pest.size)    

    """ Fetch the bracket list """
    n_ests = form_alpha_noise_est(mins,freqs,use_pickle=True)
    b_list = get_brackets(mins, n_ests, freqs, use_pickle=False)

    """ Perform a batch inversion on the first ten seconds of data """
    dm = min_dom[1]-min_dom[0]
    num_f = 5
    df = .1
    T = 100
    poly_order = 5
    batch_size = int(T*1500)
    print('batch_size',batch_size)

    theta0, Sigma, bracks =get_initial_est(b_list, num_f, df, poly_order, batch_size)
    print(theta0)
    t = np.linspace(0, T, batch_size)
    dt = 1/1500
    plt.figure()
    for x in bracks:
        plt.plot(dt*x.dom[:batch_size], x.data[:batch_size])
    batch_bracks = bracks[:] 
    batch_batch_size = batch_size
    

    """ Perform a sequential inversion on the first ten seconds, using
    the first 5 seconds to seed to 10 second inversion """
    T = 2.5
    batch_size=int(T*1500)
    theta0, Sigma, bracks =get_initial_est(b_list, num_f, df, batch_size, poly_order)
    """ Filter out bracks that are opened"""
    new_b_list = [x for x in b_list if x not in bracks]
    """ reset start indices to batch_size """
    start_ind = batch_size
    """ Filter out bracks that are closed """
    bracks = [x for x in bracks if x.end > start_ind]
    
    tmp = time.time() 
    theta, Sigma, alpha_list, curr_ind, open_bracks, leftover_b_list = seq_least_squares(new_b_list, num_f, df, poly_order, theta0, Sigma, start_ind, batch_size, bracks)
    x = time.time() - tmp
    print('time for seq', x)
    print('predicted total time is ', (33.5*60 / 2.5 * x / 60), ' hours')
    print('sls theta')
    print(theta)


    bracks_used = [x for x in b_list if x.start < batch_size]
    double_check = [x for x in b_list if (x in bracks) or (x in open_bracks)]
    double_check = list(set(double_check))
    double_check = sort_by_start(double_check)
    batch_bracks = sort_by_start(batch_bracks)

    plt.figure()
    for x in double_check:
        plt.plot(dt*x.dom[:batch_batch_size], x.data[:batch_batch_size])
    plt.show()


def run_long_sls():

    """ Perform a sequential inversion on the 5 minutes, using
    the first 5 seconds to seed the sequential thing """
    pest = get_pest(49, 1)
    sec_dom = np.linspace(0, (40-6.5-dm)*60, pest.size)    

    """ Fetch the bracket list """
    n_ests = form_alpha_noise_est(mins,freqs,use_pickle=True)
    b_list = get_brackets(mins, n_ests, freqs, use_pickle=False)

    """ PIckl num model params"""
    num_f = 5
    df = .1
    poly_order = 4
    

    """ Run initial batch estimate """
    T = 40
    batch_size=int(T*1500)
    theta0, Sigma, bracks =get_initial_est(b_list, num_f, df, poly_order, batch_size)
    print('initial ests', theta0)

    """ Filter out bracks that are opened"""
    new_b_list = [x for x in b_list if x not in bracks]
    """ reset start indices to batch_size """
    start_ind = batch_size

    """ Select the length of the sequential inversion """
    t_min = 32
    num_samples = t_min*60*1500
    """ Filter out bracks that are closed """
    bracks = [x for x in bracks if x.end > start_ind]
    tmp = time.time() 
    theta, Sigma, alpha_list, curr_ind, open_bracks, leftover_b_list = seq_least_squares(new_b_list, num_f, df, poly_order, theta0, Sigma, start_ind, num_samples, bracks)
    x = time.time() - tmp
    print('time for seq', x)
    print('sls theta')
    print(theta)

    with open('pickles/sls_poly_result.pickle', 'wb') as f:
        pickle.dump([theta, np.diag(Sigma), curr_ind, open_bracks, leftover_b_list],f)

def compute_chunk_estimate(chunk_len):
    """
    Perform an estimate alpha dot for a series of chunks of length chunk_len(seconds)
    Break data up into chunks
    Within each chunk, move the bracket's starting phase measurements to 0
    to remove the offset errs. 
    Then perform a small batch inversion to get things started, finish the
    chunk with sls
    chunk_len - float
        lenght of each chunk in seconds
    """
    pest = get_pest(49, 1)
    sec_dom = np.linspace(0, (40-6.5-dm)*60, pest.size)    

    """ Fetch the bracket list """
    n_ests = form_alpha_noise_est(mins,freqs,use_pickle=True)
    b_list = get_brackets(mins, n_ests, freqs, use_pickle=False)
    
    """ get lenght of pest records """
    total_len = pest.size

    """ Get number of chunks. It throws out some data at the end """
    chunk_size = 1500*chunk_len
    num_chunks = int(total_len // chunk_size)
    theta_chunks = []
    sigma_chunks = []

    """ open the first brackets """
    b_list = sort_by_start(b_list)
    open_bracks = []
    i = 0
    while b_list[i].start == 0:
        x = b_list.pop(0)
        x.add_alpha0(0)
        _,_ = x.get_data()
        open_bracks.append(x)

    for chunk in range(num_chunks):
        start_ind = chunk_size*chunk
        """
        Set all the open bracks to start their data at phi0 = 0
        """
        for x in open_bracks:
            x.add_alpha0(0)
            """ I can't use the routine """
            x0= x.data[0]
            x.data -= x0

        """ Use a bullshit mesmt with huge covariance to start sls"""
        theta, Sigma = np.array([1]).reshape(1,1), np.matrix([1e6])

        """ Run sls for the chunk """
        theta, Sigma, alpha_list, curr_ind, open_bracks, leftover_b_list = seq_least_squares(b_list, 0, 1, 1, theta, Sigma, start_ind, batch_size=chunk_size-1, open_bracks=open_bracks, model_sec_start=0)

        """ Update b_list """
        b_list = leftover_b_list

        """ add estimates ot record """
        theta_chunks.append(theta[0,0])
        sigma_chunks.append(Sigma[0,0])
        print('-----------finished sls for chunk', chunk)
        print(theta)
        print(curr_ind, chunk_size*(chunk+1))

    return theta_chunks, sigma_chunks
        
        


def look_at_long_run():    
    with open('pickles/sls_result.pickle', 'rb') as f:
        theta, Sigma, alpha_list, curr_ind, open_bracks, leftover_b_list = pickle.load(f)

    print(theta)


if __name__ == '__main__':
    start = time.time()
    _ = get_pest(49,1)
    dm = 1/1500/60
    chunk_len = 5
    theta, sigma = compute_chunk_estimate(chunk_len)
    plt.plot(theta)
    plt.show()
    with open('pickles/chunk_theta_' + str(chunk_len) + '.pickle', 'wb') as f:
        pickle.dump([theta, sigma], f)
    #compare_sls_to_batch()
    #run_long_sls()
    #look_at_long_run()

    end = time.time()
    print('total time', end-start)
