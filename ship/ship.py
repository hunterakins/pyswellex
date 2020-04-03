import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy.stats import mode
from scipy.io import loadmat
from scipy.interpolate import interp1d


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
