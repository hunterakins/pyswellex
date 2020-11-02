import numpy as np
from matplotlib import pyplot as plt
from env.env.envs import factory
import os
from pyat.pyat.readwrite import read_modes
'''
Description:
Based on my synthetic aperture preliminary results, I've seen that
certain modes show up as several different peaks. This suggests that
there are multiple k_r showing up as the same mode over the course of the
0 minute to 40 minute segment. 
I'm interested in seeing how sensitive the kr are to SSP.
The plan is to solve the modal equation for a certain frequency for all the CTD casts in
the experiment, then scatter the kr to see what the spread looks like. 
Then, I will map the spread back to an expected distribution in frequency and 
see if a hypothetical range dependence given by the variation in temporal CTD casts
can explain the frequency multiplicity of the modes.

It is worth also looking at the frequency spread and calculating the kr variation required
to obtain such a spread. 

Date: 
7/14/2020

Author: Hunter Akins
'''



env_builder = factory.create('swellex')
env = env_builder()
zr = np.linspace(94.125, 212.25, 64)
zr = np.delete(zr,21)
zs = 9
freq = 127
dz, zmax, dr, rmax = 2, 216.5, 1, 1.2*1e3
env.add_source_params(freq, zs, zr)
env.add_field_params( dz, zmax, dr, rmax)
#prefix = '/home/hunter/research/code/env/env/ctds/'
#prn_files = os.listdir(prefix)
#prn_files = [prefix + x for x in prn_files if x[-3:] == 'prn']
folder = 'at_files/'
fname = 'swell'
#for p in prn_files:
#    env.change_cw(p)
#    env.run_model('kraken', folder, fname, zr_range_flag=False)
#    modes = read_modes(**{'fname':folder+fname+'.mod', 'freq':freq})
#    krs = modes.k
#    num_modes = len(krs)
#    print('num_modes', num_modes)
#    dom = np.linspace(1, num_modes, num_modes)
#    plt.scatter(dom, krs)
#plt.show()

""" Looking at depth effect """
env.run_model('kraken', folder, fname, zr_range_flag=False)
modes = read_modes(**{'fname':folder+fname+'.mod', 'freq':freq})
krs = modes.k
print('Spread', (np.max(krs)- np.min(krs))*2.4 /(2*np.pi))
print('min spacing', np.min(abs(krs[1:] - krs[:-1]))*2.4 /(2*np.pi))
print('max spacing', np.max(abs(krs[1:] - krs[:-1]))*2.4 /(2*np.pi))
print('Original average kr', np.mean(krs))
num_modes = len(krs)
print('num_modes', num_modes)
dom = np.linspace(1, num_modes, num_modes)
fig,axes = plt.subplots(2,1)
axes[1].set_xlabel('Mode number')
axes[0].set_ylabel('kr (1/m)')
axes[0].scatter(dom, krs)
krs_orig = krs
omega_m = 2*np.pi*freq + krs_orig*2.4
print('omega_s', omega_m)
print('freq_nm', omega_m / 2/np.pi)

delta_f = -(krs[1:] - krs[:-1]) *2.4 / (2 * np.pi)

""" Look at weighted average kr """
s_strength = abs(modes.phi[0,:])
print(s_strength)
num_modes = modes.phi.shape[1]
total_stren = np.sum(s_strength)
summ = np.sum(modes.k.real*s_strength)
print(np.sum(s_strength) / total_stren)
print('weighted mean k', summ/total_stren)
plt.figure()
print((omega_m[:-1] - omega_m[1:]) / 2 /np.pi)
plt.scatter(np.linspace(0, krs.size-2, krs.size-1), (omega_m[:-1] - omega_m[1:]) / 2 /np.pi, color='r')
plt.scatter(np.linspace(0, krs.size-2, krs.size-1), delta_f)
plt.show()

D = 180
plt.suptitle('Effective of depth change on kr (D changes from 216.5 m to ' +str(D)+' m)')
env.change_depth(D)
fname = 'wonky'
zr = np.arange(94.125, D-2, 0.5)
zr = np.delete(zr,21)
dz, zmax, dr, rmax = 2, D, 1, 1.2*1e3
env.add_source_params(freq, zs, zr)
env.add_field_params( dz, zmax, dr, rmax)
env.run_model('kraken', folder, fname, zr_range_flag=False)
modes = read_modes(**{'fname':folder+fname+'.mod', 'freq':freq})
krs = modes.k
print(D, 'm average kr', np.mean(krs))
s_strength = abs(modes.phi[0,:])
num_modes = modes.phi.shape[1]
total_stren = np.sum(s_strength)
print('total strenght', total_stren)
summ = np.sum(modes.k.real*s_strength)
print('weighted mean k', summ/total_stren)
num_modes = len(krs)
dom = np.linspace(1, num_modes, num_modes)
axes[0].scatter(dom, krs)
diffs= krs_orig[:len(krs)]-krs
axes[1].scatter(dom, 2.4*diffs/2/np.pi)
axes[1].set_ylim([0, np.max(diffs)*1.1])
axes[1].set_ylabel('Change in frequency (Hz)')
plt.show()

print('LOOOOK HERE', (np.max(krs) - np.min(krs)) * 2.4 / (2*np.pi))

""" Looking at ctd effect"""
""" Conclusion is that ctd effect is minimal """

