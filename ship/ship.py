import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy.stats import mode
from scipy.signal import convolve, firwin
from scipy.io import loadmat
from scipy.interpolate import interp1d
from swellex.audio.lse import AlphaEst
import pickle


def load_track():
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

pickle_name = '../audio/pickles/chunk_10s_auto_shallow_two_freqs.pickle'
v_est= get_est(pickle_name)
v_est.thetas = -np.array(v_est.thetas).reshape(len(v_est.thetas))
v_est.get_t_grid()
v_est.add_t0(6.5*60)
plt.plot(v_est.tgrid/60, v_est.thetas)#, yerr=np.sqrt(sigma))

pickle_name = '../audio/pickles/chunk_10s_auto_deep_two_freqs.pickle'
v_est= get_est(pickle_name)
v_est.thetas = -np.array(v_est.thetas).reshape(len(v_est.thetas))
v_est.get_t_grid()
v_est.add_t0(6.5*60)
plt.plot(v_est.tgrid/60, v_est.thetas)#, yerr=np.sqrt(sigma))

pickle_name = '../audio/pickles/chunk_10s_auto_deep_201_235_283_freqs.pickle'
v_est= get_est(pickle_name)
v_est.thetas = -np.array(v_est.thetas).reshape(len(v_est.thetas))
v_est.get_t_grid()
v_est.add_t0(6.5*60)
plt.plot(v_est.tgrid/60, v_est.thetas)#, yerr=np.sqrt(sigma))


gps_range_km, lat, lon, gps_time = load_track()
gps_vel = get_velocity(gps_time, gps_range_km)

gps_vel = gps_vel[:40]
gps_time = gps_time[:40]
plt.plot(gps_time, gps_vel)
plt.legend(['Shallow source 10s chunk linear fit using 280Hz', 'Deep source 10s chunk linear fit using 338 Hz and 388Hz', 'Deep source 10s chunk using 201, 235, and 283 Hz', 'GPS estimated velocity'])
plt.ylim([-1.8, -3])
plt.show()



