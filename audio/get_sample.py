import numpy as np
from matplotlib import pyplot as plt

'''
Description:
The supercomputer has enough mem to load all 64 elements, the whole shebang...I don't. 

This is just a way to take the full arrays from the supercomputer
and write out small chunks to analyze

Date: 

Author: Hunter Akins
'''

def get_full_pest(freq):
    pest=  np.load('/home/fakins/data/swellex/' + str(freq) + '_pest.npy')
    return pest

def get_full_td(freq):
    x = np.load('/oasis/tscc/scratch/fakins/' + str(freq) + '.npy')
    return x

def get_full_pest(freq):
    pest=  np.load('/home/fakins/data/swellex/' + str(freq) + '_pest.npy')
    return pest

def get_pest_samples(start_min, end_min, freq, fname):
    start_ind = start_min*60*1500
    end_ind = end_min*60*1500
    full_est = get_full_pest(freq)
    vals = full_est[:,start_ind:end_ind]
    np.save(fname, vals)
    return

def get_td_vals(start_min, end_min, freq, fname):
    start_ind = start_min*60*1500
    end_ind = end_min*60*1500
    full_est = get_full_td(freq)
    vals = full_est[:,start_ind:end_ind]
    np.save(fname, vals)
    return

#get_pest_samples(4, 8, 127, '/oasis/tscc/scratch/fakins/data/swellex/127_pest_short.npy')

#get_td_vals(4, 8, 127, '/oasis/tscc/scratch/fakins/data/swellex/127_td_short.npy')


x = np.load('/oasis/tscc/scratch/fakins/49_ts.npy')

np.save('/oasis/tscc/scratch/fakins/49_mini.npy', x[0:1,:])


    
    

