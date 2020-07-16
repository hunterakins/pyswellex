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

def form_alpha_noise_est(brackets,use_pickle=False):
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
            dom_segs, pest_segs = form_good_alpha(freq, brackets, i+1)
            curr_noise_ests = []
            for k in range(len(pest_segs)):
                dom, seg = dom_segs[k], pest_segs[k]
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
    denom = np.square(sigma_sample) + h.T@Sh
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
    
def get_h(curr_sec, num_f, df):
    """
    Form the new model row h.T at time t=curr_sec with num-f freq comp
    Input -
        curr_sec - float
        num_f - int
            number of frequencies to include model
    Output -
    h - numpy array (column)
    """
    h = np.ones((2*num_f+1,1))
    h[0,0] = curr_sec
    """ Populate the columns with the sines and cosines """
    if num_f > 0:
        for i in range(1, num_f+1):
            vals = curr_sec*np.cos(2*np.pi*i*df*curr_sec)
            h[2*i-1,0] = vals
            vals = curr_sec*np.sin(2*np.pi*i*df*curr_sec)
            h[2*i,0] = vals
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
        self.bracket = bracket
        self.start = bracket[0] 
        self.end = bracket[1]
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

    def get_data(self):
        dom_segs, alpha_segs = form_good_alpha(self.freq, [self.bracket], self.sensor+1)
        if type(self.alpha0) != type(None):
            x0= alpha_segs[0][0]
            diff = self.alpha0 - x0
            alpha_segs[0] += diff
        self.data = alpha_segs[0]
        self.dom = dom_segs[0]
        return dom_segs[0], alpha_segs[0]

    def __repr__(self):
        return "Bracket for sensor {:d} at frequency {:d} starting at minute {:.2f} and ending at minute {:.2f}".format(self.sensor + 1, self.freq, self.bracket[0], self.bracket[1])

def form_first_H(opening_bracks, num_f, df):
    """
    Formthe model for the first batch LS estimate
    Input -
        opening_bracks - list of Brackets
        num_f - int
            number of frequencies to include model
    Output -
    H - numpy matrix
        matrix consisting of big model for all the brackets
        truncated to the smallest bracket
    """
    """find the size of the smallest bracket
        and the corresponding domain
        while we loop through, grab all the samples I will 
        need for the inverse """
    ind = np.argmin(np.array([x.bracket[1] for x in opening_bracks]))
    smallest_brack = opening_bracks[ind]
    dom,_ =  smallest_brack.get_data()
    t = dom
    dt = t[1]-t[0]
    T = t.size*dt
    """ Form mini H"""
    H = np.ones((dom.size, 2*num_f+1))
    H[:,0] = t
    """ Populate the columns with the sines and cosines """
    if num_f > 0:
        for i in range(1, num_f+1):
            vals = t*np.cos(2*np.pi*i*df*t)
            H[:,2*i-1] = vals
            vals = t*np.sin(2*np.pi*i*df*t)
            H[:,2*i] = vals
    """ Make the big H """
    big_H = np.ones((t.size*len(opening_bracks), 2*num_f+1))
    for i in range(len(opening_bracks)):
        big_H[i*t.size:(i+1)*t.size, :] = H
    return big_H

def form_first_alpha(opening_bracks):
    """
    Get a big column vector of the first
    measurements of alpha
    """
    ind = np.argmin(np.array([x.bracket[1] for x in opening_bracks]))
    smallest_brack = opening_bracks[ind]
    print('smallest', smallest_brack)
    if type(smallest_brack.dom) != type(None):
        dom,alpha =  smallest_brack.dom, smallest_brack.data
    else:
        dom, alpha = smallest_brack.get_data()
    col_size = dom.size
    alphas = np.zeros((col_size*len(opening_bracks),1))
    for i in range(len(opening_bracks)):
        brack = opening_bracks[i]
        if type(brack.data) == type(None):
            dom, alphs= brack.get_data()
        else:
            dom, alphs = brack.dom, brack.data
        alphas[i*col_size:(i+1)*col_size,0] = alphs[:col_size]
    return alphas
    
def get_H_inv(H, opening_bracks):
    """
    Use covariance of brackets and initial thing of
    H to compute H pseudo inverse
    Hpsuedo is inv(H.T@Cinv@H)@H.T@Cinv
    C is diagonal with highly structured blocksj
    """
    """first compute block size """
    ind = np.argmin(np.array([x.bracket[1] for x in opening_bracks]))
    smallest_brack = opening_bracks[ind]
    dom,alpha =  smallest_brack.get_data()
    col_size = dom.size
    """
    Compute Cinv H and H.T @ Cinv
    """
    CiH = np.zeros(H.shape)
    CiH[:,:] = H
    for i in range(len(opening_bracks)):
        brack = opening_bracks[i]
        var = brack.alpha_var
        CiH[i*col_size:(i+1)*col_size,:] *= 1/var 
    inv = np.linalg.inv(H.T@CiH)
    pinv = inv @ CiH.T
    cov =  inv
    return pinv, cov
        
def get_brackets(mins, alpha_ests, freqs, use_pickle=True):
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

def get_initial_est(b_list, num_f, df):
    """
    get estimate of theta and Sigma to seed
    the sequential least squares algorithm
    """
    opening_bracks = [x for x in b_list if x.bracket[0] == 6.5]
    H = form_first_H(opening_bracks, num_f, df)
    alphas = form_first_alpha(opening_bracks)
    Hinv, cov = get_H_inv(H, opening_bracks)
    theta0 = Hinv@alphas
    ind = np.argmin(np.array([x.bracket[1] for x in opening_bracks]))
    smallest_brack = opening_bracks[ind]
    print('smallest_brack',smallest_brack)
    end=smallest_brack.bracket[1]
    ignored_bracks = [x for x in b_list if (x.bracket[0] < end) and(x not in opening_bracks)]
    dom,alpha =  smallest_brack.get_data()
    col_size = dom.size
    #for i, b in enumerate(opening_bracks):
    #    dom, alpha= b.get_data()
    #    val = H[i*col_size:(i+1)*col_size,:]@theta0
    start_ind = col_size
    return theta0, cov, col_size

def get_new_seed(open_brackets, num_f, df, curr_min,curr_ind):
    """
    Use the output of the sequential least squares 
    to seed a new start (i.e. allow for chunking)
    So I used the set of currently open brackets
    They should have all their data popped so that the first
    data point corresponds to curr_min
    I want at least a ten second segment
    """
    """
    Filter out brackets that are too short
    """
    print(curr_min)
    brackets = [x for x in open_brackets if (x.bracket[1]-curr_min) > 1/60]
    H = form_first_H(brackets, num_f, df)
    alphas = form_first_alpha(brackets)
    Hinv, cov = get_H_inv(H, brackets)
    theta0 = Hinv@alphas
    ind = np.argmin(np.array([x.bracket[1] for x in brackets]))
    smallest_brack = brackets[ind]
    print('smallest_brack',smallest_brack)
    end=smallest_brack.bracket[1]
    dom, alpha = smallest_brack.dom, smallest_brack.data
    col_size = dom.size
    new_ind = curr_ind+col_size 
    return theta0, cov, new_ind
   
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

def seq_least_squares(b_list, num_f, df, theta,Sigma, start_ind, open_brackets=None):
    """ Run the initial batch LS. Use output to initialize the
    counter (curr_min, curr_ind), as well as the initial theta estimate
    and initial Sigma 

    """
    _ = get_pest(49, 1)
    num_samples =  _.size
    minute_dom = np.linspace(6.5,40, num_samples)
    sec_dom = np.linspace(0, (40-6.5)*60, num_samples)
    dm = minute_dom[1]-minute_dom[0]
    start_min = minute_dom[start_ind]
    print('Starting sequential least squares at ', start_min)
    start_sec = sec_dom[start_ind]
    ds = sec_dom[1] - sec_dom[0]

    """    Sort the brackets by end_min """
    sorted_b_last = sort_by_end(b_list)
    sorted_b_first = sort_by_start(b_list)

    """ Get the open brackets """
    curr_ind = start_ind+1
    curr_min = 6.5+curr_ind*dm
    curr_sec = start_sec + ds
    if type(open_brackets) == type(None):
        open_bracks = []
    """
    Remove those that are already closed
    """
    while sorted_b_last[0].bracket[1] <= start_min:
        sorted_b_last.pop(0)
    while sorted_b_first[0].bracket[0] <= start_min:
        if sorted_b_first[0] in sorted_b_last:
            x = sorted_b_first.pop(0)
            """ init data objects """
            if type(open_brackets) == type(None):
                open_bracks.append(x)
        else:
            sorted_b_first.pop(0)
    if type(open_brackets) == type(None):
        for x in open_bracks:
            x.bracket[0] = start_min
            """ Initalize data with new start min """
            _, _ = x.get_data()
    else:
        open_bracks = sort_by_end(open_brackets)
        while open_bracks[0].bracket[1] <= curr_min:
            open_bracks.pop(0)
        print('open brackets',open_bracks)
    """
    Sort open brackets by their closing time
    """
    open_bracks = sort_by_end(open_bracks)
    """
    Reset opening to start_min for all  brackets
    """
    """
    Loop through whole system and update theta
    """
    num_samples = curr_ind + 22500
    curr_alpha = 0
    offsets = []
    alpha_list = []
    while curr_ind < num_samples:
        """ Remove brackets that closed """
        if open_bracks[0].bracket[1] < curr_min:
            print('removing bracket(s), curr_min is ', curr_min)
            while open_bracks[0].bracket[1] < curr_min:
                x = open_bracks.pop(0)
                print(len(x.data))
                print('removed bracket', x)
                print('num brackets open', len(open_bracks))
        """ add brackets that have opened """
        if curr_min >= sorted_b_first[0].bracket[0]:
            print('adding new brackets at curr min ', curr_min)
            while curr_min >= sorted_b_first[0].bracket[0]:
                x= sorted_b_first.pop(0)
                print('adding bracket ', x)
                x.add_alpha0(curr_alpha)
                print('bracket min', x.bracket[0])
                """ initialize data """
                _,_ = x.get_data() 
                open_bracks.append(x)
                print('num brackets open', len(open_bracks))
            open_bracks = sort_by_end(open_bracks)
        """ Create model row h.T """
        h = get_h(curr_sec, num_f, df)
        tmp_count = 0
        for brack in open_bracks:
            alpha = brack.data[0]
            """ pop off top element of data """
            brack.data = np.delete(brack.data, 0)
            brack.dom = np.delete(brack.dom, 0)
            """ Copmute gain matrix """
            K = get_Kn(Sigma, h, brack.alpha_var)
            theta = update_theta(theta, K, alpha, h)
            Sigma = update_Sigma(Sigma, K, h)
        curr_alpha = (h.T@theta)[0]
        alpha_list.append(curr_alpha)
        curr_ind += 1
        curr_min += dm
        curr_sec += ds
    for x in open_bracks:
        x.bracket[0] = curr_min
    return theta, Sigma, alpha_list, curr_sec, curr_ind, open_bracks
            
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
    
 

if __name__ == '__main__':
    start = time.time()
    _ = get_pest(49,1)
    num_samples = _.size
    min_dom = np.linspace(6.5,40,num_samples)
    shallow_inds = [4,6,14,15,16,17,18,19,20]
    n_ests =form_alpha_noise_est(mins, use_pickle=True)
    b_list = get_brackets(mins, n_ests, freqs,use_pickle=False) 
    b_list = [x for x in b_list if x.f_ind in shallow_inds]


    num_f = 10 
    f_max = 1
    df = f_max / num_f
    """ Run a first iteration"""
#    theta, Sigma,start_ind = get_initial_est(b_list,num_f, df)
#    print('-----------first batch-------------')
#    theta, Sigma, alpha_list, curr_sec, curr_ind,open_brackets = seq_least_squares(b_list, num_f, df, theta, Sigma, start_ind)
#    with open('pickles/sls_run '+ str(curr_ind) + '.pickle', 'wb') as f: 
#        pickle.dump([theta, Sigma, alpha_list, curr_sec, curr_ind,open_brackets],f)
#
#    curr_min = min_dom[curr_ind]
    curr_ind = 37401
    curr_min = min_dom[curr_ind]
    with open('pickles/sls_run '+ str(curr_ind) + '.pickle', 'rb') as f: 
        theta, Sigma, alpha_list, curr_sec, curr_ind, open_brackets = pickle.load(f)

    print(' ------------first batch done-----')
    print('open brackets')
    print(open_brackets)
    print(curr_min)
    print(theta)

    print(' ---------')
    dm = min_dom[1]-min_dom[0]
    for bracket in open_brackets:
        dom, alpha = bracket.dom, bracket.data
        print(dom.size, alpha.size, (bracket.bracket[1]-bracket.bracket[0]) // dm)


