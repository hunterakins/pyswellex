import numpy as np
from matplotlib import pyplot as plt

'''
Description:
To look at the raw data of arcmfp

Date: 

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


""" 

I have identified that the 65th element of the .sio file is indeed
the timer
The 64th is corrupted
The 50th is inverted

With these results in mind I rewrote save_arcmfp1 in proc_swell_data.py
to a) throw out the timer b) throw out the 64th element and c) invert the 50th element. 

"""

x = np.load('/home/fakins/data/arcmfp1/arcmfp1.npy')
print(x.shape)
num_els = x.shape[0]
for i in range(num_els):
    fig = plt.figure()
    plt.plot(x[i, :15000000:10000])
    plt.savefig('pics/' + str(i) + '_arc.png')
    plt.close(fig)
#
#
#x = np.load('/oasis/tscc/scratch/fakins/data/arcmfp1/53_32768_0.0327.npy')
#print(x.shape)
#for i in range(64):
#    fig = plt.figure()
#    plt.plot(x[i, :15000000:10000])
#    plt.savefig(str(i) + '_arc53.png')
#    plt.close(fig)

    
#x = np.load('/oasis/tscc/scratch/fakins/data/arcmfp1/197_8192_0.1219.npy')
#print(x.shape)
#for i in range(63):
#    fig = plt.figure()
#    plt.plot(x[i, :15000000:10000])
#    plt.savefig(str(i) + '_arc197.png')
#    plt.close(fig)
#
    
