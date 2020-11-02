import numpy as np
from matplotlib import pyplot as plt

'''
Description:
Shallow water autocorrelation function for ship motion


Date: 06/16

Author: Hunter Akins
'''


def get_ship_v(chunk_len):
    x = np.load('npys/sw_' + str(chunk_len) +'.npy')
    return x

def plot_v(x):
    plt.figure()
    plt.plot(x)
    plt.show()

def zero_pad(x):
    length = x.size
    new_x = np.zeros(2*length)
    new_x[:length] = x[:]
    return new_x 

def get_acf(x, size, initial_sample):
    j = initial_sample
    v = np.zeros(x.shape)
    v[:] = x[:]
    acf = []
    for i in range(size):
        xx = v[i+j:i+j+size]
        xx -= np.mean(xx)
        xx /= np.sqrt(np.var(xx))
        yy = v[j:size+j]
        yy -= np.mean(yy)
        yy /= np.sqrt(np.var(yy))
        #plt.plot(xx)
        #plt.plot(yy)
        #plt.show()
        tmp = np.sum(xx*yy)
        acf.append(tmp)
    return acf


chunk_len = 2
v = get_ship_v(chunk_len)
print(len(v))
plt.figure()
plt.plot(v)
num_samps = v.size

sample_len = int(1*60/chunk_len)
acfs = np.zeros(sample_len)
plt.figure()
for j in range(5*int(60/chunk_len)):
    acf = get_acf(v, sample_len, j)
    acfs += np.array(acf)
    plt.plot(acf)

plt.figure()
plt.suptitle('velicuty')
plt.plot(v[:sample_len])
plt.plot(v[j:j+sample_len])
plt.plot(v)


plt.figure()
plt.plot(acfs)

plt.figure()
plt.plot(v[:sample_len])
plt.plot(v[100:100+sample_len])
plt.show()

