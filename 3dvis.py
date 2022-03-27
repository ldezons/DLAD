# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data
import math

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])

    def update(self, points, sem_labels, color_map):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        number_pt = sem_labels.shape[0]
        rgb = np.zeros((number_pt, 3))
        for i in range(number_pt):
            rgb[i] = color_map[sem_labels[i, 0]]
        rgb[:, 0], rgb[:, 2] = rgb[:, 2], rgb[:, 0].copy()
        self.sem_vis.set_data(points, size=3, edge_color=rgb / 255)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

if __name__ == '__main__':
    data = load_data('data/demo.p') # Change to data.p for your final submission 
    visualizer = Visualizer()
    visualizer.update(data['velodyne'][:,:3], data['sem_label'], data['color_map'])

    box_corners = np.empty((1,8,3))
    for car in data['objects']:
        h = car[8]
        l = car[9]
        w = car[10]
        box_center = np.asarray(car[11:14]).reshape(-1, 1)
        angle = car[14]
        summits = np.asarray([[w / 2, 0, l / 2], [w / 2, -h, l / 2], [w / 2, 0, -l / 2], [w / 2, -h, -l / 2],
                              [-w / 2, 0, l / 2], [-w / 2, -h, l / 2], [-w / 2, 0, -l / 2], [-w / 2, -h, -l / 2]])

        Rot = np.asarray([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        points = np.matmul(summits, Rot).T + np.tile(box_center, ((1,8)))
        points = np.concatenate((points, np.ones((1,8))))
        points = (np.matmul(np.linalg.inv(data['T_cam0_velo']), points)).T
        box_corners = np.vstack((box_corners, points[:, 0:3].reshape(1, 8, 3)))
    visualizer.update_boxes((box_corners))
    vispy.app.run()




