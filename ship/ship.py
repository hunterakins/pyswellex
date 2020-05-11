import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy.stats import mode
from scipy.signal import convolve, firwin
from scipy.io import loadmat
from scipy.interpolate import interp1d
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

def get_shall_est(pickle_name,chunk_size,c_bias_factor=1):
    """ Chunk size in seconds c_bias_factor (difference from 1500 as proportional factor"""
    with open(pickle_name, 'rb') as f:
        thetas, sigmas = pickle.load(f)
        thetas = [x[0] for x in thetas]
        thetas = np.array(thetas)
#        sigmas = np.array([x[0] for x in sigmas])
        #print(thetas)
        #print(len(thetas))
        #thetas = np.array(thetas)
        #print(thetas.shape)
    return -c_bias_factor*thetas, np.array(sigmas)


true_speed  =1491.8
#kbar_factor = .875
kbar_factor = 1
kbar_factor = .97
true_speed  *= kbar_factor
c_bias_factor=true_speed/1500
chunk_size =6
pickle_name = '../audio/pickles/chunks_shallow_batch.pickle'
thetas,sigmas = get_shall_est(pickle_name,chunk_size,c_bias_factor)
time = chunk_size*len(thetas)
theta_time = np.arange(6.5, 6.5+(len(thetas)-.01)*chunk_size/60,chunk_size/60)
print('Total time for estimates in min', time / 60)
r, lat, lon, time = load_track()
print('--------')
plt.figure()
plt.title('SwellEx Ship Track')
plt.plot(time, r)
plt.xlabel('Time (min)')
plt.ylabel('Range (km)')
plt.savefig('track.png')


velocity = get_velocity(time, r)

filt= firwin(10, .01, fs=1/chunk_size)
thetas = thetas.reshape(thetas.size)
t_mean = np.mean(thetas)
filt_thetas = convolve(thetas-t_mean, filt, mode='same')
filt_thetas += t_mean
t_tot = chunk_size*np.sum(thetas)
total_dist = t_tot/1000
r0 = 0
print(r0)
ranges = [r0]
for i in range(len(thetas)):
    r0 += chunk_size*thetas[i]
    ranges.append(r0)
ranges = np.array(ranges)
r0 = (r[7]+r[6])/2 * 1e3
ranges += r0
th_range_km = 1e-3*ranges
t_domain = np.arange(6.5, 6.5+chunk_size/60*len(thetas)+1/60 , chunk_size/60)
plt.plot(t_domain, th_range_km)
plt.legend(['GPS position', 'Phase estimate position'])
plt.show()

    
print('Theta derived total dist', total_dist)
print(chunk_size*thetas.size/60)
r_start = (r[7]+r[6])/2
print(time[39])
r_end = (.8*r[38] + .2*r[37])
print(r_end - r_start)



inds = [i for i in range(len(velocity)) if (time[i] > 6.5) and time[i] < (theta_time[-1])]
plt.figure()
plt.plot(time[inds], velocity[inds])
plt.errorbar(theta_time, thetas)
plt.xlabel('Time (mins)')
plt.ylabel('Range rate m/s')
plt.legend(['GPS Differencing derived range rate','Phase estimation derive range rate'])

plt.figure()
plt.plot(time[inds], velocity[inds])
plt.plot(theta_time, filt_thetas)
plt.xlabel('Time (mins)')
plt.ylabel('Range rate m/s')
plt.legend(['GPS Differencing derived range rate','Phase estimation derive range rate'])

plt.show()


