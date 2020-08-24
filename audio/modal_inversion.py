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
    def __init__(self, freq, shape_list, fi=[], interp_bad_el=False):
        self.freq = freq
        self.num_ests = len(shape_list)
        inds = np.argsort(fi)
        shape_list = [shape_list[x] for x in inds]
        fi = np.sort(fi)
        fi = fi[::-1]
        shape_list = shape_list[::-1]
        self.fi = fi
        zr = np.linspace(94.125, 212.25,64)
        zr = np.delete(zr, 21)
        if interp_bad_el == True:
            tmp = np.linspace(94.125, 212.25,64)
            for i in range(len(shape_list)):
                shapef_real= interp1d(zr, shape_list[i].real.reshape(zr.size))
                shapef_imag= interp1d(zr, shape_list[i].imag.reshape(zr.size))
                shape_list[i] = shapef_real(tmp) + complex(0,1)*shapef_imag(tmp)
            zr = tmp 
        self.ests = shape_list
        self.zr = zr

    def get_real_shapes(self):
        """
        Shapes are complex
        """
        real_shapes = []
        for shape in self.ests:
            angle = np.angle(shape)
            #plt.figure()
            #plt.plot(angle%(np.pi))
            #plt.show()
            ang = np.median(angle)
            real_shape = np.exp(complex(0,1)*-ang)*shape
            real_shapes.append(real_shape.real)
            #plt.figure()
            #plt.plot(real_shape.real)
            #plt.plot(shape.real)
            #plt.plot(shape.imag)
            #plt.show()
        self.real_shapes = real_shapes
        return

    def populate_phi(self):
        phis = []
        for shape in self.smooth_shapes:
            shape /= np.max(abs(shape.real))
            p_est = unwrap(shape.real,self.zr)
            phis.append(p_est)
        self.phis = phis

    def plot_modes(self, smooth=False):
        plt.figure()
        c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, x in enumerate(self.real_shapes):
            plt.plot(x.real,self.zr, color=c_list[i]) 
        for i, x in enumerate(self.smooth_shapes):
            plt.plot(x.real,self.zr, color=c_list[i]) 
            
    def get_smooth_shapes(self):
        """
        Estimated mode shapes are quite rough, so 
        do a short filter to help them """
        filt = 1/5*np.array([1, 1, 1, 1, 1])
        self.smooth_shapes = []
        for shape in self.real_shapes:
            new_shape = np.zeros(shape.shape)
            for i in range(len(filt)):
                new_shape[i] = shape[i]
                new_shape[-(i+1)] = shape[-(i+1)]
            for j in range(len(filt), shape.size-len(filt)):
                new_shape[j] = np.sum(filt*shape[j-2:j+len(filt)-2])
            self.smooth_shapes.append(new_shape)
        return

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
        c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i,phase in enumerate(phases[1:]):
            phase -= phase[5]
            ax.plot(self.zr[5:-5], phase[5:-5], color=c_list[i+1])
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Phase (radians)')
        plt.legend([str(i+1) for i in range(len(phases))])
        return fig

    def match_phi(self, shape, kz):
        """
        For given kz,
        do a global search for the phase offest that 
        matches the shape best
        """
        phis = np.linspace(0, 2*np.pi, 50)
        replica_mat = np.zeros((phis.size, shape.size))
        for i, phi in enumerate(phis):
            replica_mat[i,:] = np.cos(phi + kz*self.zr)
        diffs = replica_mat - shape
        err_size = np.linalg.norm(diffs, axis=1)
        best_phi_ind = np.argmin(err_size)
        best_phi = phis[best_phi_ind]
        return best_phi, err_size[best_phi_ind]

    def kz_estimate(self):
        """
        Perform a global search for the right 
        kz 
        """
        omega = 2*np.pi*self.freq
        best_kzs = []
        best_phis = []
        lam_max = 2*(np.max(self.zr) - np.min(self.zr))
        lam_min = (np.max(self.zr) - np.min(self.zr))/10
        kz_min = 2*np.pi / (lam_max)
        kz_max = 2*np.pi / (lam_min)
        kz_grid = np.linspace(kz_min, kz_max, 400)
        for shape in self.smooth_shapes:
            shape /= np.max(abs(shape))
            strengths = []
            phis = []
            for kz in kz_grid:
                best_phi,err_stren = self.match_phi(shape, kz)
                strengths.append(err_stren)
                phis.append(best_phi)
            best_ind = np.argmin(strengths)
            best_phi = phis[best_ind]
            best_kz = kz_grid[best_ind]
            best_kzs.append(best_kz)
            best_phis.append(best_phi)
        self.kzs = best_kzs
        self.best_phis = best_phis

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

    def show_kz_match(self, best=True):
        kzs = self.best_kzs
        phis = self.best_phis
        i = 0
        for shape in self.best_shapes:
            shape /= np.max(abs(shape))
            replica = np.cos(phis[i] + kzs[i]*self.zr)
            err = np.linalg.norm(replica-shape)
            plt.figure()
            plt.plot(shape)
            plt.plot(replica)
            plt.text(3, .8, str(err))
            i += 1
        plt.show()

    def run_inv(self):
        kzs = self.best_kzs
        fi = self.best_fi
        om_m = 2*np.pi*fi
        om = 2*np.pi*self.freq
        model_mat = make_param_model(om, om_m)
        inv = np.linalg.inv(model_mat.T@model_mat)@model_mat.T
        p = inv@np.square(kzs)
        v_hat = 1/np.sqrt(p[1])
        c_hat = 1/np.sqrt(p[0])
        if np.sign(np.median(om_m-om)) > 0:
            v_hat = -v_hat
        self.v_hat = v_hat
        self.c_hat = c_hat
        return v_hat, c_hat
   
    def compare_forward_model(self):
        kzs = self.best_kzs
        fis = self.best_fi
        om_m =(2*np.pi*self.fi)
        om = (2*np.pi*self.freq)
        model_mat = make_param_model(om, om_m)
        p = np.matrix([self.c_hat, self.v_hat])
        p = p.reshape(2,1)
        p = np.power(p, -2)
        kz_hats = model_mat@p
        plt.plot(np.square(kzs))
        plt.plot(kz_hats)
        plt.show()
        

    def remove_bad_matches(self):
        kzs = np.array(self.kzs)
        fis = np.array(self.fi)
        bad_inds = []
        const = np.square(kzs)+ np.square(fis-self.freq)
        phis = self.best_phis
        bad_inds = []
        self.best_shapes = [self.smooth_shapes[i] for i in range(len(self.smooth_shapes)) if i not in  bad_inds]
        self.best_fi = np.array([self.fi[i] for i in range(len(self.fi)) if i not in bad_inds])
        self.best_kzs = np.array([self.kzs[i] for i in range(len(self.kzs)) if i not in  bad_inds])
        self.best_phis = np.array([self.best_phis[i] for i in range(len(self.kzs)) if i not in  bad_inds])
        return

    def __repr__(self):
        return 'ModeEstimates object for frequency ' + str(self.freq) + ' with ' + str(len(self.ests)) + ' mode estimates'

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
        
def make_param_model(om, om_m):
    """
    Given knowledge of source frequency om = 2pifreq_s
    and estimates of om_m (the doppler shifted modes)
    form matrix that relates source velocity and mean ssp
    to mode shape wavenumbers kz
    kz^2 = om_m^2 / c^2 - krm^2 = om_m^2 / c^2 - (om_s^2 -om_m^2)/v^2
    """
    om_m = np.array(om_m).reshape(len(om_m),1)
    model_mat = np.zeros((len(om_m), 2))
    om_col = np.array([om]*len(om_m))
    om_col = om_col.reshape(len(om_col),1)
    #model_mat[:,0] = np.square(om_col[:,0])
    model_mat[:,0] = np.square(om_m)[:,0]
    model_mat[:,1] = -np.square((om_m - om_col)[:,0])
    return model_mat
