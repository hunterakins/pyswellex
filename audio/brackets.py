import numpy as np
from matplotlib import pyplot as plt
import pickle
from scipy.signal import detrend
import sys

'''
Description:
Store all the code to do the bracket stuff


Date: 

Author: Hunter Akins
'''

def get_name(f0, s_ind):
    """
    get the filename of loca...Fetch the pickled phase estimate corresponding to 
    frequency f0 and sensor s_ind
    s_ind - int >= 1 """
    fname = 'pickles/kay_s' + str(s_ind) + 'pest_' + str(int(f0)) + '.pickle'
    return fname

def get_pest(data_loc, s_ind):
    """
    """
#    fname = 'pickles/kay_s' + str(s_ind) + 'pest_' + str(int(f0)) + '.pickle'
    print('getting pest t ', data_loc)
    p_est = np.load(data_loc)
    p_est = p_est[s_ind-1, :]
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

def form_good_pest(f, brackets, data_loc, sensor_ind=1,detrend_flag=False):
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
    pest =get_pest(data_loc,sensor_ind)
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

def form_good_alpha(f, brackets, data_loc, sensor_ind=1, c=1500, pest=None):
    """
    alpha = c*((phi - phi(0))/(2 pi f0) - t)
    Use the list of good brackets to extract the relevant portions 
    """
    if type(pest) == type(None):
        pest =get_pest(data_loc, sensor_ind)
    pest -= pest[0] 
    dom = np.linspace(0, pest.size-1, pest.size, dtype=int)
    dm = 1/1500/60
    sec_dom = np.linspace(0, (75-dm)*60, pest.size)
    dom = np.linspace(0, pest.size-1, pest.size, dtype=int)
    ds = 1/1500
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

def  cpa_domain(bracket):
    """
    Input - 
        bracket - list of floats
            floats represent minutes in seconds
    """
    min_dom = np.arange(40, 75-1/1500/60, 1/1500/60)
    end = bracket[1]-1
    start = bracket[0]
    start_min = min_dom[start]
    end_min = min_dom[end]
    return start_min, end_min, start, end

class Bracket:
    """ Bracket of good data """
    def __init__(self, bracket, sensor_ind, freq, f_ind, data_loc, alpha_var=None):
        """ 
        Input 
            bracket - list, bracket[0] is start index, bracket[1] is end index
            in mins
            sensor_ind - int >= 1
            freq - int 
                frequency band
            f_ind - int
                index of the frequency in the ''canonical list''
            data_loc - str
                filepath of pickled data
            alpha_var - None type or float  
                the variance of the rescaled phase rate for the bracket
        """
        self.start, self.end = bracket[0], bracket[1]
        self.bracket = [self.start, self.end]
        self.sensor = sensor_ind
        self.freq = freq
        self.f_ind = f_ind
        self.data_loc = data_loc 
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

    def get_data(self,c=1500, pest=None):
        """
        Get data corresponding to the bracket
        Optionally provide the numpy array
        of phase estimates relevant for this 
        brackets frequency and sensor index """
        dom_segs, alpha_segs = form_good_alpha(self.freq, [self.bracket], self.data_loc, self.sensor, c=c,pest=pest)
        if type(self.alpha0) != type(None):
            x0= alpha_segs[0][0]
            diff = self.alpha0 - x0
            alpha_segs[0] += diff
        self.data = alpha_segs[0]
        self.dom = dom_segs[0]
        return dom_segs[0], alpha_segs[0]

    def __repr__(self):
        return "Bracket for sensor {:d} at frequency {:d} starting at minute {:.2f} and ending at minute {:.2f}".format(self.sensor, self.freq, self.start_min, self.end_min)

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

def get_cpa_brackets(mins, freqs, use_pickle=True):
    if use_pickle == True:
        with open('brackets/cpabrackets.pickle', 'rb') as f:
            brackets = pickle.load(f)
            return brackets
    else:
        b_objs = []
        for i in range(len(mins)):
            for j in range(len(freqs)):
                freq_ests = mins[i][j]
                for k,bracket in enumerate(freq_ests):
                    b_obj = Bracket(bracket, i+1, freqs[j], j,cpa=True)
                    #b_obj.add_alpha_var()
                    b_objs.append(b_obj)
        with open('brackets/cpabrackets.pickle', 'wb') as f:
            pickle.dump(b_objs, f) 
        return b_objs

def cleanup_autobrackets(brackets, use_pickle=True, cpa=False):
    """
    Some little hacky algorithm that tosses out the bad brackets
    of the automatically selected brackets
    cpa=True means that the brackets correspond to minutes
    40 to 75, cpa=False means that the brackets correspond to 6.5 to 40
    """
    if cpa == False:
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
    if cpa == True:
        if use_pickle == True:
            with open('brackets/clean_cpaautobrackets.pickle', 'rb') as f:
                brackets = pickle.load(f)
                return brackets
        freqs= [49, 64, 79, 94, 109, 112,127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    #    vrs = []
        clean_bracks = []
        for f in freqs:
            bracks = [x for x in brackets if x.freq == f]
            for j in range(21):
                sensor_bracks = [x for x in bracks if x.sensor == j+1]
                dats = detrend(get_cpa_pest(f, sens[j+1]))
                for x in sensor_bracks:
                    vals = dats[x.start:x.end]
                    tmp = detrend(vals)
                    if np.var(tmp) < .6:
                        clean_bracks.append(x)
        with open('brackets/clean_cpaautobrackets.pickle', 'wb') as f:
            pickle.dump(clean_bracks, f)
    return clean_bracks

def get_autobrackets(mins, freqs, use_pickle=True,cpa=False):
    if cpa == False:
        if use_pickle == True:
            with open('brackets/clean_autobrackets.pickle', 'rb') as f:
                brackets = pickle.load(f)
                return brackets
    else:
        with open('brackets/cpabrackets.pickle', 'rb') as f:
            brackets = pickle.load(f)
            return brackets

def get_cpa_auto_inds_in_mins():
    """ Convert the indices produced by
    the automatic segment detector to the 
    WRONG mins thing I did early, all so I can
    correct it back when I instantiate the brackets..."""
    from indices.cpasensors_auto import mins
    pest = get_cpa_pest(232, 1)
    num_samples=pest.size
    dm = 1/1500/60
    min_dom = np.linspace(40, 75-dm, num_samples)
    dmm = min_dom[1] - min_dom[0]
    print(dmm)
    min_dom = np.arange(40, 75-1/1500/60, 1/1500/60)
    for i in range(len(mins)): # sensors
        sensor_bracks = mins[i]
        num_freqs = len(sensor_bracks)
        for j in range(num_freqs):
            f_bracks = sensor_bracks[j]
            for k in range(len(f_bracks)):
                brack = f_bracks[k]
                mins[i][j][k] = [min_dom[brack[0]], min_dom[brack[1]-1]]
    return mins
    
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
    cpa_domain([40, 41])
    #freqs = [127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385]
    freqs= [49, 64, 79, 94, 109, 112,127, 130, 148, 166, 201, 235, 283, 338, 388, 145, 163, 198,232, 280, 335, 385] 
    #freqs = [201, 235, 283, 338, 388, 232, 280, 335, 385]
    #freqs = [283]


    """ Process first segment brackets """
    #mins = get_auto_inds_in_mins()
    #brackets = get_brackets(mins, freqs, use_pickle=True)
    #cleanup_autobrackets(brackets, use_pickle=True)
    #sys.exit(0)
    
    #from indices.cpasensors_auto import mins
    #brackets = get_cpa_brackets(mins, freqs, True)
    #sys.exit(0)
    #
    ##mins = get_cpa_auto_inds_in_mins() 
    #brackets = get_autobrackets(mins, freqs,cpa=True)
    ##brackets  =get_brackets(mins, freqs, False)
    ##clean_bracks = cleanup_autobrackets(brackets, False)
    #bracks = [x for x in brackets if x.freq in freqs]
    #for j in range(21):
    #    sens = j+1
    #    s_bracks = [x for x in bracks if x.sensor == sens]
    #    dats = detrend(get_cpa_pest(freqs[0], sens))
    #    for x in s_bracks[-30:]:
    #        dom,vals = x.get_data()
    #        tmp = detrend(vals)
    #        plt.plot(dom,vals,color='r')
    #    plt.show()
