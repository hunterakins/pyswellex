import numpy as np
from matplotlib import pyplot as plt

'''
Description:
Reverse the indices of the data
for running things backwards

Date: 

Author: Hunter Akins
'''

from .sensors import mins

reverse_mins = []
for sensor in mins:
    rev_sensor = []
    for freq in sensor:
        freq.reverse()
        for i in range(len(freq)):
            x = freq[i]
            x0 = x[0]
            x1 = x[1]   
            y0 = 46.5 -  x[1]
            y1 = 46.5 - x[0]
            freq[i] = [y0, y1]
        rev_sensor.append(freq)
    reverse_mins.append(rev_sensor)


       
