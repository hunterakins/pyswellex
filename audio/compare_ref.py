import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
'''
Description:


Date: 

Author: Hunter Akins
'''

freq = 109


data_dir = 'npy_files/109/ref_comp/'
ref_est = np.load(data_dir+'1.0_nb_385.npy')
curr_est = np.load(data_dir+'1.0_nb_curr.npy')
ref_f = np.load(data_dir + 'nb_385_f_grid.npy')
curr_f  = np.load(data_dir + 'nb_curr_f_grid.npy')
raw_est = np.load('npy_files/109/ref_comp/0.0_nb_curr.npy')
no_f = curr_f
ref_est, curr_est, raw_est = ref_est.T, curr_est.T, raw_est.T

def get_pow(est):
    return np.sum(abs(est), axis=0)

ref_pow = get_pow(ref_est)
curr_pow = get_pow(curr_est)
raw_pow = get_pow(raw_est)

norm = np.max(ref_pow)
ref_db = ref_pow / norm
ref_db = 10*np.log10(ref_db)
raw_db = raw_pow / norm
raw_db = 10*np.log10(raw_db)
curr_db = curr_pow / norm
curr_db = 10*np.log10(curr_db)


def get_frs(freq, f_grid):
    krs = np.load('npy_files/' + str(freq) + '_krs.npy')
    krs = krs.real[:11]
    v_est = -(freq - np.mean(f_grid)) / freq * 1500
    frs = (krs-np.mean(krs)) * v_est / 2 / np.pi + np.mean(curr_f)
    return frs

def plot_db_spec(freq, f, est, no_f, no_est):
    est_pow = get_pow(est)
    no_pow = get_pow(no_est)

    norm = np.max(est_pow)
    est_db = est_pow / norm
    est_db = 10*np.log10(est_db)
    no_db = no_pow / norm
    no_db = 10*np.log10(no_db)

    fig, ax = plt.subplots(1,1)
    plt.suptitle('Acceleration compensation effectiveness')
    ax.plot(no_f,no_db, alpha=0.5)
    ax.plot(f, est_db, color='g')

    frs = get_frs(freq, f)
    
    for fr in frs:
        ax.plot([fr]*10, np.linspace(-15, 0, 10), '-', alpha=.5, color='k')
    ax.legend(['No compensations', 'Using ' + str(freq) + 'inst. freq estimates to remove accelerations', 'Simulated locations of the wavenumber peaks using a ref. sound speed of 1500'])
    plt.ylabel('Power (dB, normalized to maximum power of all three PSDs)')
    plt.xlabel('Frequency')
    #plt.legend(['Using 385 reference', 'Using 109 reference', 'Using no reference', 'Simulated kr, shifted using an assumed sound speed of 1530 m/s'])
    return fig

fig = plot_db_spec(freq, ref_f, ref_est, no_f, raw_est)
fig.savefig('hihunt.png', dpi=500)


krs = np.load('npy_files/109_krs.npy')
krs = krs.real[:11]
#plt.plot(krs[:11])
#plt.show()
v_est = -(109 - np.mean(curr_f)) / 109 * 1500
offset = 0.0013
frs = (krs-np.mean(krs)) * v_est / 2 / np.pi + np.mean(curr_f)+offset

peak_inds, peak_heights = find_peaks(ref_db,height=-5)
#for peak_ind in peak_inds:
#    vals = ref_est[:,peak_ind]
#    plt.plot(vals.real)
#    plt.plot(vals.imag)
#    plt.show()


fig, axes = plt.subplots(2,1)
plt.suptitle('Acceleration compensation effectiveness')
axes[0].plot(no_f,raw_db, alpha=0.5)
axes[1].plot(no_f,raw_db, alpha=0.5)
axes[1].plot(ref_f, ref_db,color='g')
axes[0].plot(curr_f, curr_db, color='b')
for fr in frs:
    axes[0].plot([fr]*10, np.linspace(-15, 0, 10), '-', alpha=.5, color='k')
    axes[1].plot([fr]*10, np.linspace(-15, 0, 10), '-', alpha=.5, color='k')
#plt.scatter(ref_f[peak_inds], [-3]*peak_inds.size, marker='x', color='r')
axes[0].legend(['No compensations', 'Using 109 inst. freq estimates to remove accelerations', 'Simulated locations of the wavenumber peaks using a ref. sound speed of 1500'])
axes[1].legend(['No compensations', 'Using 385 inst. freq estimates to remove accelerations', 'Simulated locations of the wavenumber peaks using a ref. sound speed of 1500'])
plt.ylabel('Power (dB, normalized to maximum power of all three PSDs)')
plt.xlabel('Frequency')
#plt.legend(['Using 385 reference', 'Using 109 reference', 'Using no reference', 'Simulated kr, shifted using an assumed sound speed of 1530 m/s'])
plt.show()


