import numpy as np
from matplotlib import pyplot as plt

'''
Description:
Look at properties of the computed segments

Consider adding noise estimates to each bracket?

Look at the minimum of the brackets
Date: 

Author: Hunter Akins
'''

from sensors import mins
from sensor1 import freqs
minn = 100000000000000
for sensor_mins in mins:
    for freq_mins, f in zip(sensor_mins, freqs):
        diffs = [x[1] - x[0] for x in freq_mins]
        x = np.min(diffs)*60
        if x < minn:
            minn = x


