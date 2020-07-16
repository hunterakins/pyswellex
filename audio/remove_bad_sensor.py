import numpy as np
from matplotlib import pyplot as plt

'''
Description:
Sensor 21 is a bad one
We must eliminate it.

Date: 6/23/2020

Author: Hunter Akins
'''

x = np.load('/home/fakins/data/s5.npy')
x = np.delete(x, 21, 0)
np.save('/oasis/tscc/scratch/fakins/data/swellex/s5_good.npy',x)

