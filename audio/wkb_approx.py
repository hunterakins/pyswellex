import numpy as np
from matplotlib import pyplot as plt
from env.env.envs import factory
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.signal import detrend
import numpy.polynomial.legendre as le
from swellex.audio.modal_inversion import ModeEstimates,unwrap

'''
Description:
Examine the validity of the WKB approximation to depth functions
for the frequencies of interest in Swellex experiment

Date: 
7/8/2020

Author: Hunter Akins
'''

def get_ssp():
    """
    Get the sound speed on the array
    """
    env_builder  =factory.create('swellex')
    env = env_builder()
    zs = 10
    zr = np.linspace(94.125, 212.25,64)
    zr = np.delete(zr, 21)
    freq = 100
    env.add_source_params(100, zs, zr)
    print(env.z_ss.size, env.cw.size)
    z_ss = env.z_ss.reshape(env.z_ss.size)
    cw = env.cw.reshape(env.cw.size)
    c_func = interp1d(z_ss, cw)
    array_ssp = c_func(zr)
    return  array_ssp

def norm_phi(phi):
    """ normalize the mode-shapes"""
    for i in range(phi.shape[1]):
        phi[:, i] /= np.max(abs(phi[:,i]))
        phi[:,i] *= np.sign(phi[0,i])
    return phi

def load_mode_shapes(freq):
    x = np.load('npy_files/' + str(freq) + '_shapes.npy')
    return x

def load_mode_krs(freq):
    x = np.load('npy_files/' + str(freq) + '_krs.npy')
    return x

def load_mode_model(freq):
    """ Get kraken mode shapes """
    phi = load_mode_shapes(freq) 
    phi = phi.real
    phi = norm_phi(phi)
    krs = load_mode_krs(freq)
    return phi, krs

def wkb(freq, kr, ssp, mode_shape, zr):
    om = 2*np.pi*freq
    bar_k = om/1500
    eps = 1/bar_k
    delta_c = ssp - 1500
    Q = -(1 - 2*delta_c/1500 - kr*kr/bar_k/bar_k)
    """ 
    To get S0, integrate Q 
    """
    Q_func = interp1d(zr, np.sqrt(-Q))
    int_Q = np.array([0] + [quad(Q_func, zr[0], zr[i])[0] for i in range(1,len(Q))])
    S0 = 1/eps * int_Q
    

    """
    I need to also match coefficients. Since mode_shape is real
    i know c0 = c1_conj, and since i normalize the mode shapes,
    I know abs(c0) = abs(c1) = 1, so the only missing parameter is 
    just a phase term """
    p0 = mode_shape[0]
    phase = np.arccos(p0)
    """ Get right arccos branch """
    initial_slope = mode_shape[1]-mode_shape[0]
    if initial_slope > 0:
        phase = -phase
    phi_offset = phase - S0[0]
    """ Now I have to resolve the ambiguity """
    wkb_mode = np.cos(S0+phi_offset)
    return wkb_mode

def get_phi0(mode_shape):
    p0 = mode_shape[0]
    phase = np.arccos(p0)
    """ Get right arccos branch """
    initial_slope = mode_shape[1]-mode_shape[0]
    if initial_slope > 0:
        phase = -phase
    return phase
    
def invert_q(zr, mode_shape):
    """
    Represent the wkb kernel (sqrt(q)) in a 
    basis of legendre polynomials
    Then, the definite integral of sqrt(q)
    is expressed exactly
    The phase argument a0, a1, ... of a 
    normalized modal shape should be 
    sqrt(q)*k_bar
    Thus, find coefficients beta_0, beta_1, ...
    of N (number of receivers) Legendre polynomials
    such that the sequence of definite integrals 
    with ranges of integration [z0, z1], [z0, z2], ...
    yield the values a0, a1, ...
    Input-
    zr - np array
        rcvr depths
    mode_shape - np array
    """
    phi0 = get_phi0(mode_shape)
    num_unknowns = mode_shape.size-1
    series_list = [le.Legendre.basis(i, domain=[zr[0], zr[-1]]) for i in range(num_unknowns)]
    for x in series_list:
        a, b = x.linspace(100) 
        plt.plot(a, b)
        plt.show()
        
    return

def make_param_model(om, om_m):
    """
    Given knowledge of source frequency om = 2pifreq_s
    and estimates of om_m (the doppler shifted modes)
    form matrix that relates source velocity and mean ssp
    to mode shape wavenumbers kz
    """
    om_m = np.array(om_m).reshape(len(om_m),1)
    model_mat = np.zeros((len(om_m), 2))
    om_col = np.array([om]*len(om_m))
    om_col = om_col.reshape(len(om_col),1)
    model_mat[:,0] = np.square(om_col[:,0])
    model_mat[:,1] = -np.square((om_m - om_col)[:,0])
    return model_mat

def test_inv():
    freq = 127
    v = -2.4
    phi, krs = load_mode_model(freq)
    zr_inter = np.linspace(94.125, 212.25,64)
    zr = np.delete(zr_inter, 21)
    ssp = get_ssp()

    c_bar = np.mean(ssp)
    krs = krs.real
    om = 2*np.pi*freq
    """ Map the krs to the time domain """
    om_m = om - krs*v

    """ Get model """
    model_mat = make_param_model(om, om_m)

    """ Run forward model for funzies """
    params = np.array([1/c_bar/c_bar, 1/v/v]).reshape(2,1)
    kz_pred = model_mat@params

    delta_c = ssp - np.mean(ssp)
    me = ModeEstimates(freq, [phi[:,i] for i in range(len(krs))])
    me.get_kz_list() 
    kzs = me.kzs


    """ Now invert and see how we do..."""
    inv = np.linalg.inv(model_mat.T@model_mat)@model_mat.T
    p = inv@np.square(kzs)
    v_hat = 1/np.sqrt(p[1])
    c_hat = 1/np.sqrt(p[0])
    if np.sign(np.median(om_m-om)) > 0:
        v_hat = -v_hat
    print(c_hat, c_bar)
    print(v_hat, v)
   
    kr_hats = (om_m-om)/-v_hat 
    mode_num = np.linspace(0, len(kr_hats)-1, len(kr_hats))
    plt.figure()
    plt.scatter(mode_num, krs)
    plt.scatter(mode_num, kr_hats)
    plt.show()

def test():
    freq = 49
    phi, krs = load_mode_model(freq)
    zr_inter = np.linspace(94.125, 212.25,64)
    zr= np.delete(zr_inter, 21)
    ssp = get_ssp()
    plt.plot(ssp)
    plt.show()
    krs = krs.real
    om = 2*np.pi*freq
    mode_pic = 4
    args=unwrap(phi[:,mode_pic], zr)
    delta_c = ssp - np.mean(ssp)
    anom = delta_c / np.power(np.mean(ssp),3) * om*om  
    anom_integ = [0]
    for i in range(len(anom)-1):
        delta_z = zr[i+1]-zr[i]
        contrib = anom[i+1]*delta_z
        anom_integ.append(anom_integ[-1] + contrib)
    me = ModeEstimates(freq, [phi[:,i] for i in range(len(krs))])
    zmin = 100
    zmax = 200
    me.restrict_domain(zmin, zmax)
    #fig = me.plot_unwrapped()
    #plt.show()
    me.get_kz_list()
    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    fig3, ax3 = plt.subplots(1,1)
    errs = []
    for i in range(len(krs)):
        wkb_mode = wkb(freq, krs[i], ssp, phi[:,i], zr)
        #err = np.sqrt(np.var(phi[:,i]-wkb_mode))
        ax3.plot(zr, wkb_mode)
        ax3.plot(zr, phi[:,i])
        #errs.append(err)
        ax1.scatter(krs[i], me.kzs[i])
        ax2.plot(me.zr, detrend(me.phis[i]))
#    plt.legend([str(x) for x in range(len(krs))])
    #errs = np.array(errs)
    squares = np.square(me.kzs)
    ax1.set_xlabel('kr')
    ax1.set_ylabel('kz')
    fig1.suptitle('Estimated kz versus kr')
    fig2.suptitle('Deviation of phase from linear trend')
    fig3.suptitle('Comparing WKB derived mode from KRAKEN mode for '+ str(freq)+ ' Hz')
    ax2.set_xlabel('Depth')
    ax2.set_ylabel('Phase nonlinearity (radians)')
    kbars = np.sqrt(squares+ np.square(krs) )
    #plt.figure()
    #plt.xlabel('Standard deviation')
    #plt.xlabel('kr')
    #plt.plot(krs, errs)

    fig,ax = plt.subplots(1,1)
    ax.set_xlabel('kr')
    ax.set_ylabel('Mean SSP est. (m/s)')
    ax.scatter(krs, np.sqrt(om*om/kbars/kbars))
    ax.plot(krs, [np.mean(ssp)]*len(krs), '-', color='r')
    ax.legend(['Mean sound speed over array', 'Estimated mean c along array from phase of mode shape'])
    #plt.subplot(212)
    #plt.scatter(krs, errs)
    plt.show()

#test()
if __name__ == '__main__':
    test_inv()
    #test()
