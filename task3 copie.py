import numpy as np
import math as m
import os
from load_data import load_data
import matplotlib.pyplot as plt

def main():
    data = load_data('data/demo.p')
    img = data['image_2']
    lidar = data['velodyne'][:,0:3]  #Take only x y and z
    x = lidar[:,0].reshape(-1,1)
    y = lidar[:,1].reshape(-1,1)
    z = lidar[:,2].reshape(-1,1)
    T = data['T_cam0_velo']
    P = data['P_rect_20']
    n = lidar.shape[0]

    angles = np.degrees(np.arcsin(z/np.sqrt(x*x + y*y + z*z)))
    minth = angles.min()
    maxth = angles.max()
    print(angles.shape)
    ID = np.digitize(angles,np.linspace(minth,maxth,65))

    ID_mod = ID % 4 # because we use 4 colors
    color = np.array(['#0000FF',    #blue
                      '#FFFF00',  #yellow
                      '#FF0000',    #red
                      '#00FF00'])   #green

    color_arr = []                  
    for i in range(n):
        color_arr.append(color[ID_mod[i]])

    lidar_h = np.concatenate((lidar,np.ones(ID.shape)),axis=1)

    cam_pts = (P @ T @ lidar_h.T).T
    cam_pts_rescaled = cam_pts/cam_pts[:,2].reshape(-1,1)
    mask     = x>0
    not_mask = x<=0
    x_img = cam_pts_rescaled[:,0].reshape(-1,1)
    y_img = cam_pts_rescaled[:,1].reshape(-1,1)
    color_img = np.array(color_arr)[mask].reshape(-1,1)

    plt.scatter(x_img[mask],y_img[mask],s=0.3,c=color_img.flatten())
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()