import matplotlib.pyplot as plt
import os
import numpy as np
import math
import data_utils as d

def main():
    #We import lidar's data
    data_path = os.path.join('data/problem_4/velodyne_points/data/0000000037.bin')
    lidar = d.load_from_bin(data_path)

    x_lidar = lidar[:,0].flatten()
    y_lidar = lidar[:,1].flatten()
    z_lidar = lidar[:,2].flatten()
    lidar_length = x_lidar.shape[0]

    #We project lidar data to the camera
    R, T = d.calib_velo2cam('data/problem_4/calib_velo_to_cam.txt')
    cam_coord = np.matmul(R, lidar.T) + np.tile(T, (1, lidar.shape[0]))

    #from cam0 to cam2
    P = d.calib_cam2cam('data/problem_4/calib_cam_to_cam.txt', '02')
    cam2 = np.matmul(P, np.vstack((cam_coord, np.ones(np.shape(cam_coord)[1]))))
    cam2 = cam2/ cam2[2, :].reshape(1,-1)

    x_cam = (cam2.T[lidar[:, 0]>0][:, 0])
    y_cam = (cam2.T[lidar[:, 0]>0][:, 1])

    #we compute de distance between the points and the  car to associate each point to the correct color
    dist = np.sqrt(x_lidar * x_lidar + y_lidar * y_lidar + z_lidar * z_lidar).reshape(-1,1)
    color = d.depth_color(dist, dist[lidar[:, 0]>0].min(), dist[lidar[:, 0]>0].max())[lidar[:, 0]>0].reshape(-1,1)
    image = plt.imread('data/problem_4/image_02/data/0000000037.png')

    # Uncorrected plot
    im_proj = d.print_projection_plt(np.vstack((x_cam, y_cam)), color, (image * 255).astype(np.uint8))
    plt.imshow(im_proj)
    plt.show()

    
    # We then add the motion distortion
    velocity = d.load_oxts_velocity('data/problem_4/oxts/data/0000000037.txt')
    angular_rate_f, angular_rate_l, angular_rate_u = d.load_oxts_angular_rate('data/problem_4/oxts/data/0000000037.txt')
    t0 = d.compute_timestamps('data/problem_4/velodyne_points/timestamps_start.txt', 37)
    tf = d.compute_timestamps('data/problem_4/velodyne_points/timestamps_end.txt', 37)
    t_cam = d.compute_timestamps('data/problem_4/velodyne_points/timestamps.txt', 37)

    print(t_cam, t0, tf)
    # velodyne to imu referential
    R_imu_to_velo, T_imu_to_velo = d.calib_velo2cam('data/problem_4/calib_imu_to_velo.txt')
    imu_coord = np.matmul(np.linalg.inv(R_imu_to_velo), lidar.T - np.tile(T_imu_to_velo, (1, lidar.shape[0])))

    x_imu = imu_coord[0, :]
    y_imu = imu_coord[1, :]
    z_imu = imu_coord[2, :]

    imu_ext = np.vstack((imu_coord, np.ones(len(x_imu))))
    angle = np.arctan(-y_lidar / (x_lidar + np.finfo(float).eps))
    T_tot = tf - t0
    dt = (angle / (2 * np.pi)) * T_tot
    coord_rect = np.zeros((np.shape(imu_ext)))
    for i in range(lidar_length):
        teta = dt[i] * angular_rate_u
        dx = velocity[0] * dt[i]
        dy = velocity[1] * dt[i]
        dz = velocity[2] * dt[i]
        Def = np.array([[math.cos(teta), -math.sin(teta), 0, dx],
                      [math.sin(teta), math.cos(teta), 0, dy],
                      [0, 0, 1, dz], [0, 0, 0, 1]])
        coord_rect[:, i] = np.matmul(Def, imu_ext[:, i])
    coord_rect = coord_rect.T
    # Rectified coordinates in IMU frame
    x_rect = (coord_rect[:, 0] / coord_rect[:, 3])
    y_rect = (coord_rect[:, 1] / coord_rect[:, 3])
    z_rect = (coord_rect[:, 2] / coord_rect[:, 3])

    lidar_rect = np.vstack((x_rect, y_rect, z_rect))
    lidar_rect = np.matmul(R_imu_to_velo, lidar_rect) + np.tile(T_imu_to_velo, (1, lidar.shape[0]))
    x_rect = lidar_rect[0, :]
    y_rect = lidar_rect[1, :]
    z_rect = lidar_rect[2, :]
    depth_ = np.sqrt(x_rect * x_rect + y_rect * y_rect + z_rect * z_rect)

    # Then we project onto the image and plot the depth colored points
    print(T)

    cam_coord_rect = np.matmul(R, lidar_rect) + np.tile(T, (1, lidar_rect.shape[1]))

    cam2_rect = np.matmul(P, np.vstack((cam_coord_rect, np.ones(np.shape(cam_coord_rect)[1]))))
    cam2_rect = cam2_rect / cam2_rect[2, :].reshape(1, -1)

    x_cam_rect = (cam2_rect.T[lidar[:, 0] > 0][:, 0])
    y_cam_rect = (cam2_rect.T[lidar[:, 0] > 0][:, 1])

    color_ = d.depth_color(dist, depth_[lidar[:, 0] > 0].min(), depth_[lidar[:, 0] > 0].max())[
        lidar[:, 0] > 0].reshape(-1, 1)

    # Corrected plot

    im_proj_rect = d.print_projection_plt(np.vstack((x_cam_rect, y_cam_rect)), color_, (image * 255).astype(np.uint8))
    plt.imshow(im_proj_rect)
    plt.show()


if __name__ == '__main__':
    main()

