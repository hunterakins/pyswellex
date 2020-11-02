import numpy as np
from matplotlib import pyplot as plt
import os 
'''
Description:
Read in prn files for swellex data set
into nice objects

Author: Hunter Akins
'''


def parse_prn(fname):
    """
    Read in prn file and extract depth and ssp info
    Input:
    fname : string
    Output:
    depth, ssp : tuple of 1-d numpy arrays
    """
    depths, ssp = [], []
    with open(fname) as f:
        for line in f.readlines():
            vals = line.split(',')[1:-1]
            depth, ss = float(vals[0]), float(vals[3])
            depths.append(depth)
            ssp.append(ss)
    depths = np.array(depths).reshape(len(depths))
    ssp = np.array(ssp).reshape(len(ssp))
    return depths, ssp

def parse_swellex_ctds(root_dir):
    """
    Read all swellex .prn files into a nested list of numpy arrays
    """
    zs, ssps = [], []
    files = os.listdir(root_dir)
    files = [root_dir + '/' + x for x in files if x[-3:] == 'prn']
    for f in files:
        z, c = parse_prn(f)
        zs.append(z)
        ssps.append(c)
    return zs, ssps

def get_eofs_top(zs, ssps):
    """
    Only get eofs from top 120 meters of data. 
    It's a compromise between coverage of water column and number of available
    measurements at depth. 
    """
    top_zs = [x for x in zs if x[-1] >= 120]
    top_ssps = [x for x,y in zip(ssps,zs) if y[-1] >= 120]
    # truncate at 120 meters
    top_ind = [i for i in range(len(top_zs[0])) if top_zs[0][i] == 120][0]
    top_zs = [x[:top_ind+1] for x in top_zs]
    top_ssps = [x[:top_ind+1] for x in top_ssps]

    avg_ssp = sum(top_ssps)/len(top_ssps)
    top_delta_c = [x - avg_ssp for x in top_ssps]


    # impose a tapered function
    # with e-folding of 20 meters
    tail = np.exp(-np.arange(0, 20, .5)/10)
    tail = tail.reshape(len(tail),1)
    num_profs = len(top_delta_c)
    array_dc = np.zeros((len(top_delta_c[0]), num_profs))
    for i in range(num_profs):
        array_dc[:,i] = top_delta_c[i].reshape(len(top_delta_c[i]))
    array_dc[-40:,:] = tail*array_dc[-40:,:]
    [u,s,vh] = np.linalg.svd(np.cov(array_dc))
    return u,s, array_dc

def get_res_matrix(u,s, dropoff):
    """
    Pick out the meaningful eofs, return them in a matrix
    input:
    u : np array
        P matrix of diagonalized covariance matrix
    s : np array
        1-d array of eigenvalues of cov (square of the variance)
    dropoff : number (float)
        ratio of dropoff in variance at which we truncate the basis
    return u_res, s_res numpy arrays
    """
    cutoff_ind = [i for i in range(len(s)) if s[i]/s[0] < dropoff][0]
    print(cutoff_ind)
    u_res = u[:,:cutoff_ind]
    s_res = s[:cutoff_ind]
    return u_res, s_res


if __name__ == '__main__':
    z, ssp = parse_prn('/home/hunter/data/swellex/ctds/i9605.prn')
    print(1/np.mean(1/ssp))
    print(np.mean(ssp))
    z_inds = [i for i in range(len(z)) if z[i] > 94.125 and z[i] < 216]
    print(np.mean(ssp[z_inds]))
#    print(ssp)
    plt.plot(ssp, z)
    plt.gca().invert_yaxis()
    plt.show()
    plt.plot(ssp[z_inds], z[z_inds])
    plt.gca().invert_yaxis()
    plt.show()
    zs, ssps = parse_swellex_ctds('/home/hunter/data/swellex/ctds')
    u,s, array_dc = get_eofs_top(zs, ssps)
    num_depths, num_msmts = np.shape(array_dc)
    ur, sr = get_res_matrix(u,s, 1e-2)
    for i in range(num_msmts):
        plt.plot(array_dc[:,i])
        plt.plot(ur@ur.T@array_dc[:,i])
        plt.show()
