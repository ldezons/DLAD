
import matplotlib.pyplot as plt
import os
import numpy as np
from load_data import load_data
def main():
    data_path = os.path.join('data', 'demo.p')
    data = load_data(data_path)

    int_trans = data['T_cam0_velo']
    nor_trans = data['P_rect_20']
    image = data['image_2']
    velo = data['velodyne']
    sem_label = data['sem_label']
    color_map = data['color_map']
    cars = data['objects']
    nb_points = velo.shape[0]

    rectified_velo = (nor_trans@int_trans@velo.T).T #Rectifying the image
    rectified_velo_scaled = rectified_velo/rectified_velo[:,2].reshape(-1,1) #Scaling the image by dividing by the z-component

    xline = rectified_velo_scaled[velo[:,0]>0][:,0] #Splitting the data in two differents vectors x and y while only keeping the values with x>0
    yline = rectified_velo_scaled[velo[:,0]>0][:,1]

    coloring = np.zeros((nb_points,3))
    for i in range(nb_points):
        coloring[i] = color_map[sem_label[i, 0]]

    coloring = coloring[velo[:,0]>0]/255
    for car in cars:
        draw_box(car,nor_trans)
    plt.scatter(xline, yline,s=0.3,c=coloring)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def draw_box(car, nor_trans):
    h = car[8]
    l = car[9]
    w = car[10]
    rot = car[14]
    rot_mat = np.array([[np.cos(rot) ,0      ,np.sin(rot)], #Rotation about y axis
                       [0           ,1                ,0],
                       [-np.sin(rot),0      ,np.cos(rot)]])

    ccoor = np.array(car[11:14]).reshape(-1,1) #Car Coordinates in camera 0 coordinates

    boundary = np.array([[w/2, 0, l/2], [w/2, -h, l/2], [w/2, 0 , -l/2], [w/2, -h, -l/2],
                      [-w/2, 0, l/2], [-w/2, -h, l/2], [-w/2, 0 , -l/2], [-w/2, -h, -l/2]] )

    box = np.tile(ccoor, (1,8))  + rot_mat@ boundary.T #Translating the boundary by the car coordinates
    hom_box = np.vstack((box,np.ones((1,8)))) #Transforming it to homogeneous coord
    box_cam = nor_trans @ hom_box
    print(box_cam[2,:].reshape(1,-1))
    box_cam_scaled = box_cam/box_cam[2,:].reshape(1,-1)

    box_plot(box_cam_scaled, 0, 1)
    box_plot(box_cam_scaled, 0, 2)
    box_plot(box_cam_scaled, 0, 4)
    box_plot(box_cam_scaled, 1, 3)
    box_plot(box_cam_scaled, 1, 5)
    box_plot(box_cam_scaled, 2, 3)
    box_plot(box_cam_scaled, 2, 6)
    box_plot(box_cam_scaled, 3, 7)
    box_plot(box_cam_scaled, 4, 5)
    box_plot(box_cam_scaled, 4, 6)
    box_plot(box_cam_scaled, 5, 7)
    box_plot(box_cam_scaled, 6, 7)


def box_plot(box_cam_scaled, p1, p2):

    plt.plot([box_cam_scaled[0, p1], box_cam_scaled[0, p2]], [box_cam_scaled[1, p1], box_cam_scaled[1, p2]], color='#39FF14')

if __name__ == '__main__':
    main()