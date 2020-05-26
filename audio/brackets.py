import numpy as np
from matplotlib import pyplot as plt
import pickle
from scipy.signal import detrend

'''
Description:
Store all the code to do the bracket stuff


Date: 

Author: Hunter Akins
'''

def get_pest(f0, s_ind):
    """
    Fetch the pickled phase estimate corresponding to 
    frequency f0 and sensor s_ind
    s_ind - int >= 1
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
        List of the left index and right index of
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
    dom = np.linspace(0, pest.size, pest.size)
    inds = []
    size = 0
    dom_segs = []
    pest_segs = []
    for bracket in brackets:
        if bracket[0] > bracket[1]:
            print(f, bracket)
            raise(ValueError, 'bracket isn\'t increasing')
        li = bracket[0]
        ri = bracket[1]
        seg = pest[li:ri]
        dom_seg = dom[li:ri]
        pest_segs.append(seg)
        dom_segs.append(dom_seg)
    return dom_segs, pest_segs

def form_good_alpha(f, brackets, sensor_ind=1, c=1500):
    """
    alpha = c*((phi - phi(0))/(2 pi f0) - t)
    Use the list of good brackets to extract the relevant portions 
    """
    pest =get_pest(f, sensor_ind)
    pest -= pest[0] 
    dom = np.linspace(0, pest.size-1, pest.size, dtype=int)
    dm = 1/1500/60
    sec_dom = np.linspace(0, (40-6.5-dm)*60, pest.size)
    inds = []
    d_segs = []
    alpha_segs = []
    for bracket in brackets:
        if bracket[0] > bracket[1]:
            print(f, bracket)
            raise(ValueError, 'bracket isn\'t increasing')
        li = bracket[0]
        ri = bracket[1]
        seg = pest[li:ri]
        dom_seg = dom[li:ri]
        """ Get time in seconds """
        t_seg = sec_dom[li:ri]
        alpha_seg= c*(seg / (2*np.pi*f) - t_seg)
        alpha_segs.append(alpha_seg)
        d_segs.append(dom_seg)
    return d_segs, alpha_segs

def correct_bad_domain(bracket):
    """
    When I wrote the script to select the 
    indices of the good portions, I accidentally
    truncated the domain by one sample, so the 
    minutes referenced in the brackets .py files
    in indices/ are different from the actual 
    times by a factor of 1.0000003316612551
    Rather than correct them in the files, 
    I correct them here """
    dm = 1/1500/60
    """ Convert bracket values to the true minutes """ 
    start_min = 6.5 + (bracket[0]-6.5)/1.0000003316612551
    end_min = 6.5 + (bracket[1]-6.5)/1.0000003316612551
    """ index of start and end of bracket """
    start = int((start_min-6.5) / dm)
    end = int((end_min-6.5) / dm)
    return start_min, end_min, start, end

class Bracket:
    """ Bracket of good data """
    def __init__(self, bracket, sensor_ind, freq, f_ind, alpha_var=None):
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
        start_min, end_min, start, end = correct_bad_domain(bracket)
        """ Minutes corresponding to first and last sample """
        self.start_min, self.end_min = start_min, end_min
        """ index of start and end of bracket (in the domain (0, pest.size, pest.size)"""
        self.start, self.end = start, end
        self.bracket = [self.start, self.end]
        self.sensor = sensor_ind
        self.freq = freq
        self.f_ind = f_ind
        self.alpha_var = alpha_var
        self.alpha0=None
        self.dom = None
        self.data = None

    def add_alpha_var(self):
        if type(self.data) == type(None):
            _,_ = self.get_data()
            seg = detrend(self.data)
            noise_var = np.var(seg)
            self.alpha_var = noise_var
            """ Clear data to conserve mem """
            self.data, self.dom = 0,0

    def add_alpha0(self, alpha0):
        """
        add an alpha0
        """
        self.alpha0 = alpha0

    def get_data(self,c=1500):
        dom_segs, alpha_segs = form_good_alpha(self.freq, [self.bracket], self.sensor, c)
        if type(self.alpha0) != type(None):
            x0= alpha_segs[0][0]
            diff = self.alpha0 - x0
            alpha_segs[0] += diff
        self.data = alpha_segs[0]
        self.dom = dom_segs[0]
        return dom_segs[0], alpha_segs[0]

    def __repr__(self):
        return "Bracket for sensor {:d} at frequency {:d} starting at minute {:.2f} and ending at minute {:.2f}".format(self.sensor, self.freq, self.start_min, self.end_min)

#def form_noise_est(brackets):
#    """
#    Get sample variance of detrended "good segments"
#    For list of these sample variances to use as estimates of the noise 
#    variance for each sensor/frequency
#    Input -
#    brackets- list of list of list of list
#        Each element is a list corresponding to a sensor
#        Each sensor has a list corresponding to each freq
#        Each freq has a list of 2-element lists
#        The first element is the start minute of the good seg,
#        second element is the end minute
#    Output -
#        noise_ests - list of list of lists
#        noise_ests[0][0][0] is the variance around a line of the
#        first element at 49 Hz for the first good segment of data
#    """
#    noise_ests = []
#    for i, sensor_bracket in enumerate(brackets):
#        freq_ests = []
#        for j, freq in enumerate(freqs):
#            brackets = sensor_bracket[j]
#            dom_segs, pest_segs = form_good_pest(freq, brackets, i+1,detrend_flag=True)
#            curr_noise_ests = []
#            for seg in pest_segs:
#                seg = detrend(seg)
#                noise_var = np.var(seg)
#                curr_noise_ests.append(noise_var)
#            freq_ests.append(curr_noise_ests)
#        noise_ests.append(freq_ests)
#    return noise_ests
#
#def form_alpha_noise_est(brackets,freqs,use_pickle=False,c=1500):
#    """
#    Get sample variance of detrended "good segments" in the alpha
#    variable
#    For list of these sample variances to use as estimates of the noise 
#    variance for each sensor/frequency
#    Input -
#    brackets- list of list of list of list
#        Each sensor has a list of lists corresponding to each freq
#        Each freq has a list of 2-element lists
#        The first element is the start minute of the good seg,
#        second element is the end minute
#    Output -
#        noise_ests - list of list of lists
#        noise_ests[0][0][0] is the variance around a line of the
#        first element at 49 Hz for the first good segment of data
#    """
#    if use_pickle==True:
#        with open('noise/alpha_noise.pickle','rb') as f:
#            est = pickle.load(f)
#            return est
#    noise_ests = []
#    for i, sensor_bracket in enumerate(brackets):
#        freq_ests = []
#        for j, freq in enumerate(freqs):
#            brackets = sensor_bracket[j]
#            dom_segs, pest_segs = form_good_alpha(freq, brackets, i+1,c)
#            curr_noise_ests = []
#            for k in range(len(pest_segs)):
#                dom, seg = dom_segs[k], pest_segs[k]
#                if dom.size == 0:
#                    print(brackets)
#                seg = detrend(seg)
#                noise_var = np.var(seg)
#                curr_noise_ests.append(noise_var)
#            freq_ests.append(curr_noise_ests)
#        noise_ests.append(freq_ests)
#    with open('noise/alpha_noise.pickle', 'wb') as f:
#        pickle.dump(noise_ests, f)
#    return noise_ests

def get_brackets(mins, freqs, use_pickle=True):
    if use_pickle == True:
        with open('brackets/brackets.pickle', 'rb') as f:
            brackets = pickle.load(f)
            return brackets
    else:
        b_objs = []
        for i in range(len(mins)):
            for j in range(len(freqs)):
                freq_ests = mins[i][j]
                for k,bracket in enumerate(freq_ests):
                    b_obj = Bracket(bracket, i+1, freqs[j], j)
                    #b_obj.add_alpha_var()
                    b_objs.append(b_obj)
        with open('brackets/brackets.pickle', 'wb') as f:
            pickle.dump(b_objs, f) 
        return b_objs

def cleanup_autobrackets(brackets, use_pickle=True):
    if use_pickle == True:
        with open('brackets/clean_autobrackets.pickle', 'rb') as f:
            brackets = pickle.load(f)
            return brackets
    freqs= [49, 64, 79, 94, 109, 112,127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    dom = np.linspace(6.5, 40, 3015000)
#    vrs = []
    clean_bracks = []
    for f in freqs:
        bracks = [x for x in brackets if x.freq == f]
        for j in range(21):
            data = get_pest(f, j+1)
            detrend(data, overwrite_data=True)
            sensor_bracks = [x for x in bracks if x.sensor == j+1]
            sensor_bracks = [x for x in sensor_bracks if data[x.end] > data[x.start]]
            mean_slope = np.mean(np.array([data[x.end] - data[x.start] for x in sensor_bracks]))
            sensor_bracks = [x for x in sensor_bracks if ((data[x.end] - data[x.start]) < 2*mean_slope)] 
            print(mean_slope)
            for x in sensor_bracks:
                clean_bracks.append(x)
    with open('brackets/clean_autobrackets.pickle', 'wb') as f:
        pickle.dump(clean_bracks, f)
    return clean_bracks

def get_autobrackets(mins, freqs, use_pickle=True):
    if use_pickle == True:
        with open('brackets/clean_autobrackets.pickle', 'rb') as f:
            brackets = pickle.load(f)
            return brackets
    
def get_auto_inds_in_mins():
    """ Convert the indices produced by
    the automatic segment detector to the 
    WRONG mins thing I did early, all so I can
    correct it back when I instantiate the brackets..."""
    from indices.sensors_auto import mins
    min_dom = np.linspace(6.5, 40,3015000) 
    for i in range(len(mins)): # sensors
        sensor_bracks = mins[i]
        num_freqs = len(sensor_bracks)
        for j in range(num_freqs):
            f_bracks = sensor_bracks[j]
            for k in range(len(f_bracks)):
                brack = f_bracks[k]
                mins[i][j][k] = [min_dom[brack[0]], min_dom[brack[1]-1]]
    return mins
           


if __name__ == '__main__':
    #freqs = [127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385]
    freqs= [49, 64, 79, 94, 109, 112,127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    mins = get_auto_inds_in_mins() 
    brackets  =get_brackets(mins, freqs, False)
    clean_bracks = cleanup_autobrackets(brackets)
    bracks = [x for x in clean_bracks if x.freq == 127]
    for j in range(21):
        sens = j+1
        s_bracks = [x for x in bracks if x.sensor == sens]
        dats = detrend(get_pest(127, sens))
        for x in s_bracks:
            plt.plot(dats[x.start:x.end])
        plt.show()
