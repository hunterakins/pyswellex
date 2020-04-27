import numpy as np
from matplotlib import pyplot as plt
from indices.sensors import freqs, mins
from proc_phase import form_good_pest, form_good_alpha
from scipy.signal import detrend
import pickle

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

def get_Kn(Sigma, h, sigma_curr):
    """
    Get the sequential least squares gain matrix
    K from Sigma, the current error covariance matrix,
    h, the newest model row as a column matrix!, and sigma_curr, the estimated
    noise on the newest measurement
    Return 
    Kn the gain matrix
    """ 
    Sh = Sigma@h
    numer = Sh
    denom = np.square(sigma_curr) + h.T@Sh
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
    Sigma_new = (np.identity - K@h.T)@Sigma_curr
    return Sigma_new
    
def test_sls():
    """
    Test out the sequential least squares algorithm
    """
    
    return


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
    def get_data(self):
        dom_segs, alpha_segs = form_good_alpha(self.freq, [self.bracket], self.sensor+1)
        return dom_segs[0], alpha_segs[0]
    def __repr__(self):
        return "Bracket for sensor {:d} at frequency {:d} starting at minute {:.2f} and ending at minute {:.2f}".format(self.sensor + 1, self.freq, self.bracket[0], self.bracket[1])

def form_first_H(opening_bracks, num_f):
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
    df = 1/T
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
    dom,alpha =  smallest_brack.get_data()
    col_size = dom.size
    alphas = np.zeros((col_size*len(opening_bracks),1))
    for i in range(len(opening_bracks)):
        brack = opening_bracks[i]
        dom, alphs= brack.get_data()
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

            

def get_initial_est(b_list, num_f):
    """
    get estimate of theta and Sigma to seed
    the sequential least squares algorithm
    """
    opening_bracks = [x for x in b_list if x.bracket[0] == 6.5]
    H = form_first_H(opening_bracks, num_f)
    alphas = form_first_alpha(opening_bracks)
    Hinv, cov = get_H_inv(H, opening_bracks)
    theta0 = Hinv@alphas
    ind = np.argmin(np.array([x.bracket[1] for x in opening_bracks]))
    smallest_brack = opening_bracks[ind]
    dom,alpha =  smallest_brack.get_data()
    col_size = dom.size
    for i, b in enumerate(opening_bracks):
        dom, alpha= b.get_data()
        plt.plot(dom[:col_size], alpha[:col_size])
        val = H[i*col_size:(i+1)*col_size,:]@theta0
        plt.plot(dom[:col_size], val)
    plt.show()
    return theta0, cov
    

def seq_least_squares(b_list, num_f):
    theta, Sigma,start_ind = get_initial_est(b_list,num_f)
    return
 

if __name__ == '__main__':
#    shallow_inds = [4,6,14,15,16,17,18,19,20]
    n_ests =form_alpha_noise_est(mins, use_pickle=True)
    b_list = get_brackets(mins, n_ests, freqs,use_pickle=False) 

    opening_bracks = [x for x in b_list if x.bracket[0] == 6.5]
    for b in opening_bracks:
        dom, alpha = b.get_data()
        plt.plot(dom, alpha)
    plt.show()
    num_f = 20
    seq_least_squares(b_list, num_f)
 
    
