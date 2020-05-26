import os
import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.indices.sensors import freqs, mins
from swellex.audio.proc_phase import form_good_pest, form_good_alpha, get_pest
from scipy.signal import detrend
import pickle
import time
from swellex.audio.brackets import Bracket, get_brackets, get_autobrackets


'''
Description:
Perform the least square estimation of the ship velocity

Date: 
4/26


Author: Hunter Akins
'''

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

def get_row_sizes(opening_bracks, batch_size,start_ind=0):
    max_ind = batch_size+start_ind
    num_rows = 0 # to count total num
    row_sizes = [] # to hold the size for each bracket
    for x in opening_bracks:
        dom, alpha = x.get_data()
        """ If the bracket is fully contained
        in the batch chunk, the whole data segment
        will be used """
        if x.start > start_ind:
            if x.end > max_ind:
                num_samps = max_ind - x.start
            else:
                num_samps = x.end-x.start
        else:
            if x.end > max_ind:
                num_samps = max_ind-start_ind
            else:
                num_samps = x.end-start_ind
        num_rows+=num_samps
        row_sizes.append(num_samps)
    return num_rows, row_sizes
  
def form_first_H(opening_bracks, num_f, df, poly_order, batch_size,start_ind=0):
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
    max_ind = start_ind+batch_size
    num_rows, row_sizes = get_row_sizes(opening_bracks, batch_size,start_ind)
  
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
        if curr_brack.start < start_ind:
            data_start_ind = 0
        else:
            data_start_ind = curr_brack.start-start_ind
        if curr_brack.end > max_ind:
            data_end_ind = batch_size
        else:
            data_end_ind = curr_brack.end-start_ind
        row_size = row_sizes[i]
        relevant_H = H[data_start_ind:data_end_ind,:]
        big_H[curr_row:curr_row+row_size] = relevant_H
        curr_row += row_size
    return big_H

def form_first_alpha(opening_bracks, batch_size, start_ind=0):
    """
    Get a big column vector of the first
    measurements of alpha
    """
    max_ind = batch_size
    """ First determine size of column vec and 
    contribution of each bracket """
    num_rows, row_sizes = get_row_sizes(opening_bracks, batch_size,start_ind)

    alphas = np.zeros((num_rows,1))
    curr_row = 0
    for i in range(len(opening_bracks)):
        x = opening_bracks[i]
        row_size = row_sizes[i]
        if x.start > start_ind:
            data_start_ind = 0
        else:
            data_start_ind = start_ind - x.start
        y = alphas[curr_row:curr_row+row_size, 0]
        alphas[curr_row:curr_row+row_size,0] = x.data[data_start_ind:data_start_ind+row_size]
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
    end_ind = start_ind + batch_size
    opening_bracks = [x for x in b_list if (((x.start < start_ind) and (x.end > start_ind)) or ((x.start >= start_ind) and (x.start < end_ind)))]

    """ Form a model matrix from these brackets """
    H = form_first_H(opening_bracks, num_f, df, poly_order, batch_size,start_ind)

    """ COmbine the msmts from these brackets into one ''supervector'' """
    alphas = form_first_alpha(opening_bracks, batch_size,start_ind)

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
    curr_alpha = 0
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
    b_list = get_brackets(mins, freqs, use_pickle=False)

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
    b_list = get_brackets(mins, freqs, use_pickle=False)

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

def lin_offset_alpha(brack, start_ind, chunk_size,recalc_var=False):
    """
    Take a bracket that contains data in the 
    interval start_ind to start_ind + chunk_size
    Calculate the linear fit back to the origin of it then offset 
    it so that it passes the the origin at start_ind
    Input - 
    brack -Bracket object
    start_ind - int
        index of the full domain that marks the start of the
        chunk under consideration
    chunk_size - int
        number of samples in the chunk
    recalc_var - Bool   
        whether or not to recompute the variance 
        of the bracket based only
        on the values in the chunk under 
        consideration
    """

    """ If the bracket straddles the start index
    you can just set alpha0 so that it intersects
    0 at start_ind """
    _,y = brack.get_data()
    if brack.start <= start_ind:
        data_ind_start = start_ind - brack.start
        ds = brack.data[data_ind_start]
        d0 = brack.data[0]
        alpha0 = d0 - ds
        brack.add_alpha0(alpha0)
        _, y = brack.get_data()
        if recalc_var == True:
            if brack.end < start_ind + chunk_size:
                data_ind_end = -1
            else:
                data_ind_end = start_ind+chunk_size
            seg = y[data_ind_start:data_ind_end]
            if len(seg) > 1500:
                seg = detrend(seg)
                alpha_var = np.var(seg)
                brack.alpha_var = alpha_var
        return 

    """ Otherwise, estimate the slope of the data within the 
    chunk. Then pick alpha0 so that if I were to strethc
    the data backward, it would hit 0 """
    if brack.end >= start_ind + chunk_size:
        data_end = start_ind + chunk_size - brack.start
    else:
        data_end = brack.end-brack.start # the full length
    size = data_end
    h = np.linspace(0, size-1, size)
    h = h.reshape(h.size,1)
    dats = brack.data[:data_end]
    """ Get slopeof line """
    d0 = dats[0]
    dats -= d0
    if size == 0:
        print(brack)
        return
    if (h.T@h)[0,0] == 0:
        print(brack)
        return
    alpha = h.T@dats / ((h.T@h)[0,0])
    """ Now pick the offset to make it run through zero at 
    the beginning of the chunk """
    projected_initial_value = alpha*(brack.start-start_ind)
    alpha0 = projected_initial_value
    brack.add_alpha0(alpha0)
    _,y = brack.get_data()
    if recalc_var == True:
        if len(dats) > 1500:
            dats = detrend(dats)
            alpha_var = np.var(dats)
            brack.alpha_var = alpha_var
    return

def compute_chunk_estimate(chunk_len, pickle_name, freqs, recalc_var=True, recomp_bracks=False):
    """
    Perform an estimate alpha dot for a series of chunks of length chunk_len(seconds)
    Break data up into chunks
    Within each chunk, move the bracket's starting phase measurements to 0
    to remove the offset errs. 
    Then identify all brackets that will appear in the chunk, offset them all appropriately,
    and then do a batch inverse
    chunk_len - float
        lenght of each chunk in seconds
    """
    pest = get_pest(49, 1)
    sec_dom = np.linspace(0, (40-6.5-dm)*60, pest.size)    

    """ Fetch the bracket list """
#    b_list = get_brackets(mins, freqs, use_pickle=(not recomp_bracks))
    b_list = get_autobrackets(mins, freqs) # this fetches all the brackets

    b_list = [x for x in b_list if x.freq in freqs]
    poly_order = 1
    num_f = 0 # no fourier fitting
    df = 1 # dummy val
    
    """ get length of pest records """
    total_len = pest.size

    """ Get number of chunks. It throws out some data at the end """
    chunk_size = int(1500*chunk_len)
    num_chunks = int(total_len // chunk_size)-1
    theta_chunks = []
    sigma_chunks = []

    
    for chunk in range(num_chunks):
        start_ind = chunk_size*chunk
        print('start ind', start_ind, start_ind/1500/60 + 6.5)
        end_ind = start_ind + chunk_size


        """ These brackets will be used """
        opening_bracks = [x for x in b_list if (((x.start < start_ind) and (x.end > start_ind)) or ((x.start >= start_ind) and (x.start < end_ind)))]


        if len(opening_bracks) == 0:
            print('no open brackets')
            obj = theta_chunks[-1]
            theta0 = np.zeros(obj.shape)
            theta0 = np.zeros(obj.shape)
        else:
            """ Decide if you want to recalc. the variance """
            for x in opening_bracks:
                lin_offset_alpha(x, start_ind, chunk_size,recalc_var=recalc_var)
            
            theta0, cov, opening_bracks = get_initial_est(b_list, num_f, df, poly_order, chunk_size, start_ind=start_ind)
        theta_chunks.append(theta0)
        sigma_chunks.append(cov)
        alpha_est = AlphaEst(chunk_len, theta_chunks, sigma_chunks)
        with open(pickle_name, 'wb') as f:
            pickle.dump(alpha_est, f)

        """ close out brackets that won't appear again to conserve mem"""
        b_list = sort_by_end(b_list)
        while b_list[0].end <= start_ind:
            x = b_list.pop(0)
            x.dom = 0
            x.data = 0    
    return theta_chunks, sigma_chunks
        
def look_at_long_run():    
    with open('pickles/sls_result.pickle', 'rb') as f:
        theta, Sigma, alpha_list, curr_ind, open_bracks, leftover_b_list = pickle.load(f)

    print(theta)

def look_at_chunk_both():
    with open('goddamn.txt', 'r') as f:
        lines = f.readlines()
        thetas = []
        for i in range(len(lines)):
            if i % 4 == 2:
                val = lines[i][2:-3]
                print(val)
                thetas.append(float(val))
        plt.plot(thetas)
        plt.show()

def look_at_chunk_shallow():
    with open('pickles/chunks_shallow_batch.pickle', 'rb') as f:
        thetas, sigmas = pickle.load(f)
        thetas = [x[0] for x in thetas]
        plt.plot(thetas)
        plt.show()

class AlphaEst:
    def __init__(self, chunk_len, theta_list, sigma_list):
        """
        chunk_len - float
            length of the least square chunk estimates in sec
        theta_list - list of numpy ndarrays
            the parameter estimates 
        sigma_list - list of numpy ndarrays 
            parameter covariance     
        """
        self.chunk_len = chunk_len
        self.thetas = theta_list
        self.sigmas = sigma_list

    def get_t_grid(self):
        """ Make the time domain of the samples """
        num_samples = len(self.thetas)
        self.tgrid= np.linspace(0, (num_samples-1)*self.chunk_len, num_samples)
        return self.tgrid

    def add_t0(self, t0):
        """
        Take in a t0 (in seconds) and offset tgrid """
        self.tgrid += t0
        return
        
        

if __name__ == '__main__':
    start = time.time()
    _ = get_pest(49,1)
    dm = 1/1500/60
    chunk_len = 10
    #look_at_chunk_both()
    #shallow_freqs = [232, 280, 335, 385]
    #shallow_freqs = [280]
    deep_freqs = [201, 235, 283]#, 338, 388]
    #pickle_name = 'pickles/chunk_10s_auto_shallow_280Hz.pickle'
    pickle_name = 'pickles/chunk_10s_auto_deep_201_235_283_freqs.pickle'
    theta, sigma = compute_chunk_estimate(chunk_len, pickle_name, deep_freqs, recalc_var=True, recomp_bracks=False)

    end = time.time()
    print('total time', end-start)
