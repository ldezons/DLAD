import matplotlib.pyplot as plt
import os
import numpy as np
from load_data import load_data

#We define this function as we need to identify to which laser belongs each point.
# Therefore we divide the angles split the angles between 64 segments and every point has to be identified to one of these segments.
def classify(x):
    h = np.linspace(x.min(), x.max(), 64)
    c = np.zeros((len(x), 1))
    #print(np.argmin(x),x.min())
    for i in range(len(x)):
        d = False
        j = 0
        while not d:
            if h[j + 1] >= x[i] and x[i] > h[j]:
                c[i] = j+1
                d = True
                #print(c[i])
            elif x[i] == h[j]:
                c[i] = j
                d = True
            else:
                j = j+1
    return c


def main():

    data_path = os.path.join('data', 'data.p')
    data = load_data(data_path)
    int_trans = data['T_cam0_velo']
    nor_trans = data['P_rect_20']
    image = data['image_2']
    velo = data['velodyne']
    nb_points = velo.shape[0]

    x = velo[:,0].reshape(-1,1)
    y = velo[:,1].reshape(-1,1)
    z = velo[:,2].reshape(-1,1)
    #Phase of each point
    theta = np.arctan(z/np.sqrt(x**2+y**2+z**2))*180/np.pi
    #Belonging of the points
    nb = classify(theta).astype(int)

    #Modulo 4 as we use only 4 colors as done in the example
    laser_ID = nb % 4

    color = np.array(['#FF00FF',             #magenta
                          '#FF0000',         #red
                             '#00FF00',      #green
                                 '#0000FF']) #blue
    color_arr = []
    #Each point has a color
    for i in range(nb_points):
        color_arr.append(color[laser_ID[i]])

    lidar_h = np.hstack((velo[:, 0:3], np.ones((len(x), 1))))
    #print(lidar_h.shape)
    #Transformations
    transf = np.matmul(nor_trans, np.matmul(int_trans, lidar_h.T)).T
    transf_final = transf/transf[:, 2].reshape(-1, 1)
    x_final = transf_final[:,0].reshape(-1, 1)
    y_final = transf_final[:,1].reshape(-1,1)
    color_img = np.array(color_arr)[x>0].reshape(-1,1)
    plt.scatter(x_final[x>0],y_final[x>0],s=0.05,c=color_img.flatten())
    plt.imshow(image)
    plt.axis('off')
    plt.savefig("task3.jpeg")
    plt.show()

if __name__ == '__main__':
    main()