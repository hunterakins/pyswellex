import numpy as np
from matplotlib import pyplot as plt

'''
Description:

Date: 
Parse swellex track?

Author: Hunter Akins
'''

with open('swell.txt','r') as f:
    lines = f.readlines()
    times = []
    lats = []
    lons = []
    for line in lines:
        vals = line.split(' ')
        time = vals[0]
        
        lat_deg = float(vals[3])
        lat_left = float(vals[4])
        lon_deg = float(vals[7])
        lon_left = float(vals[8])
        lat, lon = lat_deg+lat_left/60, lon_deg+lon_left/60
        lats.append(lat)
        lons.append(lon)
        times.append(time)

x,y=32 + 40.254/60, 117+21.620/60

difflat, difflon = np.array([lat - x for lat in lats]), np.array([lon - y for lon in lons])
plt.scatter(difflat, difflon)
plt.show()
phim = np.array([(lat+x)/2 for lat in lats])
phim = (lats[0]+x)/2
K1 = 111.13209 - 0.56605*np.cos(2*np.pi*2*phim) + .00120*np.cos(2*np.pi*4*phim)
K2 = 111.41513*np.cos(2*np.pi*phim) - 0.09455*np.cos(2*np.pi*3*phim) + .00012*np.cos(2*np.pi*5*phim);
rangekm = np.sqrt(np.square(K1*difflat) + np.square(K2*difflon))
plt.plot(rangekm)
plt.show()

diffx, diffy = np.array(diffx), np.array(diffy)
r = np.sqrt(np.square(diffx), np.square(diffy))
plt.plot(r)
plt.show()



phim = (lat1 + lat2) / 2

K1 = 111.13209 - 0.56605*cosd(2*phim) + .00120*cosd(4*phim);
K2 = 111.41513*cosd(phim) - 0.09455*cosd(3*phim) + .00012*cosd(5*phim);
rangekm = sqrt(power(K1*difflat, 2) + power(K2*difflon,2));
