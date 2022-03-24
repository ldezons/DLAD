import matplotlib.pyplot as plt
import os
import numpy as np
from load_data import load_data

data_path = os.path.join('data', 'demo.p')
data = load_data(data_path)

int_trans = data['T_cam0_velo']
nor_trans = data['P_rect_20']
image = data['image_2']
velo = data['velodyne']
nb_points = velo.shape[0]

x = velo[:,0].reshape(-1,1)
y = velo[:,1].reshape(-1,1)
z = velo[:,2].reshape(-1,1)

theta = np.arctan(z/np.sqrt(x**2+y**2+z**2))*180/np.pi
print(theta.shape)
ID = np.digitize(theta,np.linspace(theta.min(),theta.max(),65))
ID_mod = ID % 4 # because we use 4 colors
color = np.array(['#0000FF',    #blue
                      '#FFFF00',  #yellow
                      '#FF0000',    #red
                      '#00FF00'])   #green
color_arr = []
for i in range(nb_points):
    color_arr.append(color[ID_mod[i]])
lidar_h = np.concatenate((velo[:,0:3],np.ones(ID.shape)),axis=1)
print(np.concatenate((velo[:,0:3],np.ones(ID.shape)),axis=1))
cam_pts = (nor_trans @ int_trans @ lidar_h.T).T
cam_pts_rescaled = cam_pts/cam_pts[:,2].reshape(-1,1)
x_img = cam_pts_rescaled[:,0].reshape(-1,1)
y_img = cam_pts_rescaled[:,1].reshape(-1,1)
color_img = np.array(color_arr)[x>0].reshape(-1,1)
plt.scatter(x_img[x>0],y_img[x>0],s=0.05,c=color_img.flatten())
plt.imshow(image)
plt.axis('off')
plt.show()