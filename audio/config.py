import numpy as np
from matplotlib import pyplot as plt


'''
Description:

Date: 

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

proj_ids = ['s5_deep', 's5_shallow', 'arcmfp1', 'sim1', 'sim2', 'sim3', 'arcmfp1_source15']

def get_proj_root(proj_string):
    proj_root = '/oasis/tscc/scratch/fakins/data/'
    if proj_string not in proj_ids:
        raise ValueError('Unsupported project id', proj_string)
    else:
        if proj_string[:2] == 's5':
            proj_root += 'swellex/'
        if proj_string == 'arcmfp1':
            proj_root += 'arcmfp1/'
        if proj_string == 'arcmfp1_source15':
            proj_root += 'arcmfp1_source15/'
        if proj_string == 'sim1':
            proj_root += 'sim1/'
        if proj_string == 'sim2':
            proj_root += 'sim2/'
        if proj_string == 'sim3':
            proj_root += 'sim3/'
    return proj_root

def get_full_ts_name(proj_string):
    proj_root = get_proj_root(proj_string)
    if proj_string[:2] == 's5':
        name = proj_root + 's5_good.npy'
    if proj_string == 'arcmfp1':
        name = proj_root + 'arcmfp1.npy'
    if proj_string == 'arcmfp1_source15':
        name = proj_root + 'arcmfp1.npy'
    if proj_string[:3] == 'sim':
        name = proj_root + 'sim_data.npy'
    return name

def get_proj_zr(proj_string):
    """
    Get corresponding numpy array of element
    depths for the project 
    """
    if proj_string=='s5':
        zr = np.linspace(94.125, 212.25, 64)
        zr = np.delete(zr, 21)
    elif (proj_string == 'arcmfp1') or (proj_string == 'arcmfp1_source15'):
        zr = np.linspace(94.125 + 1.875, 212.25, 63)
    elif proj_string[:3] == 'sim':
        zr = np.linspace(94.125, 212.25, 64)
    else:
        raise ValueError('unsupport projected string')
    return zr

def get_proj_tones(proj_string):
    if proj_string == 's5_deep':
        freqs = [49, 64, 79, 94,  112,130,148,166,  201, 235, 283,  338,  388]
    if proj_string == 's5_shallow':
        freqs = [109, 127, 145,163,198, 232,280, 335, 385]
    if proj_string == 'arcmfp1':
        freqs = [53, 69, 85, 101, 117, 133, 149, 165, 181, 197]
    if proj_string == 'arcmfp1_source15':
        freqs = [73, 153, 278, 373, 548, 748]
    if proj_string[:3] == 'sim':
        freqs = [49, 64]
    return freqs

def get_fc(freq, proj_string):
    """
    Get the center doppler shifted frequency
    So this can be made way more complicated...
    we're gonna just assume a velocity around
    2.4 (so this will only work on the first 35 mins of data or so)
    I account for possible errors by increasing the bandwidth
    I can refine this later once I perform the frequency estimations
    """
    if proj_string[:2] == 's5':
        fc = freq*(1 + 2.4/1500)
    if (proj_string == 'arcmfp1') or (proj_string == 'arcmfp1_source15'):
        fc= freq*(1 - 2.5/1500)
    if proj_string[:3] == 'sim':
        fc = freq*(1 - 3/1500)
    return fc

if __name__ == '__main__':
    proj_ids = ['s5_deep','s5_shallow', 'arcmfp1', 'sim1', 'sim2', 'sim3', 'arcmfp1_source15']
    print('Project ids are \'s5\' and \'arcmfp1\'. Time series are stored at ')
    print([get_full_ts_name(x) for x in proj_ids])
