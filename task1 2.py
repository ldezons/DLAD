import os
from load_data import load_data
import numpy as np
from matplotlib import transforms
from PIL import Image as im

data_path = os.path.join('data','data.p')
data = load_data(data_path)

def main ():
    velo = data['velodyne']
    shape = velo.shape
    maxi = np.amax(velo, axis=0)
    mini = np.amin(velo, axis=0)

    dist1 = maxi[0] - mini[0]   #length of the measured field along x
    dist2 = maxi[1] - mini[1]   #length of the measured field along y
    a = maxi[0]
    b = maxi[1]

    x1 = int(dist1/0.2)     #number of pixels needed to achieve a 0.2m resolution along x
    y1 = int(dist2/0.2)     #number of pixels needed to achieve a 0.2m resolution along y
    bev = np.zeros((x1 + 1, y1 + 1))

    #FILLING THE ARRAY "BEV"
    for i in range (0,shape[0]):
        v = int(y1 * (b - velo[i, 1]) / dist2)  #u,v: pixel coordinates
        u = int(x1 * (a - velo[i, 0]) / dist1)  #signs according to the veodyne coordinate system
        intensity = int(256 * velo[i, 3])
        if intensity > bev[u,v]:
            bev[u,v] = intensity

    a = np.asarray(bev)

    image = im.fromarray(a)
    image = image.convert("L")
    image.rotate(-90,expand=True).show()
    #image.save("task1.jpeg")

if __name__ == '__main__':
    main()