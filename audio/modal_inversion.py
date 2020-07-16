import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from swellex.audio.kay import diff_seq, undiff
import  numpy.polynomial.legendre as le


'''
Description:
Using mode shapes to invert for SSP

Date: 
7/6

Author: Hunter Akins
'''


def get_phi0(mode_shape):
    p0 = mode_shape[0]
    phase = np.arccos(p0)
    """ Get right arccos branch """
    initial_slope = mode_shape[1]-mode_shape[0]
    if initial_slope > 0:
        phase = 2*np.pi - phase
    return phase
  
def getextrema(vals):
    """
    find the indices of the local extrema in vals
    """ 
    diffs = vals[1:] - vals[:-1]
    extrema = []
    last_sign = np.sign(diffs[0])
    for i in range(len(diffs)):
        curr_sign = np.sign(diffs[i])
        if curr_sign != last_sign:
            last_sign=curr_sign
            extrema.append(i)
    return extrema

def unwrap(mode_shape, zr):
    """ Klugy unwrapping for 
    a cos(phi(z) + phi0) model
    Keep track of which arccos branch to 
    use by identifying the locations of the maxima"""
    p0 = get_phi0(mode_shape)
    p0 = np.arccos(mode_shape[0])
    p_est = [p0]
    for i in range(1,len(mode_shape)):
        tmp = np.arccos(mode_shape[i])
        diff = tmp -(p_est[-1]%(2*np.pi))
        p_est.append(p_est[-1]+diff)
    
    """ Now go through and stitch the branches together"""
    p_est = np.array(p_est)
    extrema = getextrema(p_est)
    if len(extrema) == 0:
        slope = np.sign(mode_shape[1] - mode_shape[i-1])
        if slope >0:
            p_est = 2*np.pi - p_est
            return p_est
        else:
            return p_est
    slopes = [np.sign(mode_shape[i] - mode_shape[i-1]) for i in extrema]
    slopes.append(np.sign(mode_shape[extrema[-1]+1] - mode_shape[extrema[-1]]))
    last_ind = 0
    for i in range(len(extrema)):
        ex_ind = extrema[i]
        rel_phase_section = p_est[last_ind:ex_ind+1]
        rel_slope = slopes[i]
        if rel_slope > 0: 
            rel_phase_section = 2*np.pi - rel_phase_section
        p_est[last_ind:ex_ind+1] = rel_phase_section
        last_ind = ex_ind+1
    """ Handle last one """
    rel_phase_section = p_est[last_ind:]
    rel_slope = slopes[-1]
    if rel_slope > 0: 
        rel_phase_section = 2*np.pi - rel_phase_section
    p_est[last_ind:] = rel_phase_section

    """ Smooth out jumps """
    for i in range(1,len(p_est)):
        val = p_est[i]
        last_val = p_est[i-1]
        if abs(val - last_val) > np.pi:
            p_est[i:] += 2*np.pi
    
    """ Interpolate the extrema (I think it's because arccos is unsstable near there?
     but not really sure what's going on)""" 
    for x in extrema:
        est = p_est[x]
        if (x < 3) or (x>(len(p_est)-4)):
            if (x == 1):
                pass
            elif (x == 2):
                vals = [p_est[i] for i in [0, 1, 3, 4]]
                z = [zr[i] for i in [0, 1, 3, 4]]
                interp = interp1d(z, vals)
                new_vals = interp(zr[2])
                p_est[2] = new_vals
            elif (x == len(p_est)-2):
                vals = [p_est[i] for i in [-3, -1]]
                z = [zr[i] for i in [-3, -1]]
                interp = interp1d(z, vals)
                new_vals = interp(zr[-2])
                p_est[-2] = new_vals
            elif (x == len(p_est)-3):
                vals = [p_est[i] for i in [-5, -4, -2, -1]]
                z = [zr[i] for i in [-5, -4, -2, -1]]
                interp = interp1d(z, vals)
                new_vals = interp(zr[-3])
                p_est[-3] = new_vals
        else:
            vals = [p_est[i] for i in [x-3, x-2, x+2, x+3]]
            z = [zr[i] for i in [x-3, x-2, x+2, x+3]]
            interp = interp1d(z, vals)
            new_vals = interp(zr[x-1:x+2])
            p_est[x-1:x+2] = new_vals
    #plt.scatter(zr[extrema], p_est[extrema], color='r')
    return p_est

def unwrap2(mode_shape):
    un = hilbert(mode_shape)
    plt.plot(un.real)
    plt.plot(un.imag)
    plt.plot(mode_shape)
    plt.show()
    yn, zn = diff_seq(un)
    ddp = np.angle(zn) 
    p0 = np.angle(un[0])
    p_est = undiff(p0, ddp, yn)
    return p_est

class ModeEstimates:
    def __init__(self, freq, shape_list, interp_bad_el=False):
        self.freq = freq
        self.num_ests = len(shape_list)
        self.ests = shape_list
        zr = np.linspace(94.125, 212.25,64)
        zr = np.delete(zr, 21)
        if interp_bad_el == True:
            tmp = np.linspace(94.125, 212.25,64)
            for i in range(len(shape_list)):
                shapef= interp1d(zr, shape_list[i].reshape(zr.size))
                shape_list[i] = shapef(tmp)
            zr = tmp 
        self.zr = zr
        phis = []
        for shape in self.ests:
            p_est = unwrap(shape,zr)
            phis.append(p_est)
        self.phis = phis
        depths = np.linspace(94.125, 212.25, 64)

    def get_kz_list(self):
        """
        Fit the phi estimates to a line least squares
        """
        zr = self.zr - self.zr[0]
        H = zr.reshape(len(zr),1) 
        Hinv = 1/(H.T@H)*H.T
        kzs = []
        for phi in self.phis:
            phi.reshape(len(phi),1)
            phi -= phi[0]
            kz = Hinv@phi
            kzs.append(kz[0])
        self.kzs = np.array(kzs)

    def plot_unwrapped(self):
        phases = self.phis
        fig,ax = plt.subplots(1,1)
        for phase in phases:
            ax.plot(self.zr, phase)
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Phase (radians)')
        return fig

    def restrict_domain(self, zmin, zmax):
        """
        Restrict the estimates to a subdomain from
        zmin to zmax """
        zr = self.zr
        if zmin < np.min(zr):
            raise ValueError('zmin is out of domain')
        if zmax > np.max(zr):
            raise ValueError('zmax is out of domain')
        ind1 = np.argmin([abs(zmin - x) for x in zr])
        ind2 = np.argmin([abs(zmax - x) for x in zr])
        self.zr = zr[ind1:ind2]
        self.phis = [x[ind1:ind2] for x in self.phis]
        self.ests = [x[ind1:ind2] for x in self.ests]

    def build_mode_mat(self):
        mat = np.zeros((self.ests[0].size, len(self.ests)))
        for i in range(len(self.ests)):
            mat[:,i] = self.ests[i][:]
        self.mode_mat = mat

def build_model_mat(est_list):
    """
    For a given set of mode shape estimates,
    which may span multiple frequencies,
    defined on the full 63 element array,
    produce the model matrix relating 
    kz^2 to sound speed and kr
    """
    num_freqs = len(est_list)
    num_krs = sum([x.num_ests for x in est_list])
    num_depths = 63
    H = np.zeros((num_depths*num_krs, num_depths + num_krs))
    """ For each frequency """
    for i, freq_est in enumerate(est_list):
        freq = freq_est.freq
        start_mat =num_depths * sum([x.num_ests for x in est_list[:i]])
        kr_offset = sum([x.num_ests for x in est_list[:i]])
        """ For each mode shape """
        for j, est in enumerate(freq_est.ests):
            start_block = j*num_depths+start_mat
            omeg = 2*np.pi*freq
            for k in range(0, num_depths):
                H[start_block+k, k] = omeg
                H[start_block+k, kr_offset + num_depths + j] = -1
    return H


def build_data_mat(est_list):
    """
    Take in the mode shapes
    which provide msmt of kz
    Form a big column matrix to go with the
    model mat above, stacking the kz estimates
    of all the freqs and shapes
    """
    num_freqs = len(est_list)
    num_krs = sum([x.num_ests for x in est_list])
    zr = est_list[0].zr
    
    return data
        
    
     

