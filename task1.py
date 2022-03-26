import os
from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from PIL import Image


def scale_to_255(a, min, max, dtype=np.uint8):

    return (((a - min) / float(max - min)) * 255).astype(dtype)


def bird_eye_view(x, y, z, intensity, side_range, fwd_range, res, min_height, max_height):
    x_lidar = np.transpose(x)
    y_lidar = np.transpose(y)
    z_lidar = np.transpose(z)
    intensity = np.transpose(intensity)
    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices] / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (x_lidar[indices] / res).astype(np.int32)  # y axis is -x in LIDAR
    # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0] / res))
    y_img -= int(np.floor(fwd_range[0] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=intensity[indices],
                           a_min=0,
                           a_max=1)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values, min=0, max=1)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[-y_img, x_img] = pixel_values  # -y because images start from top left

    # Convert from numpy array to a PIL image
    im = Image.fromarray(im)

    # SAVE THE IMAGE
    im.rotate(-90, expand=True).show()

def main():
    data_path = os.path.join('data', 'demo.p')
    data = load_data(data_path)
    x = []
    y = []
    z = []
    intensity = []
    for i in data['velodyne']:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
        intensity.append(i[3])
    plt.plot(x, y, '.', markersize=0.1)
    plt.show()
    x_lidar = np.array(x)
    y_lidar = np.array(y)
    z_lidar = np.array(z)
    intensity = np.array(intensity)
    bird_eye_view(x_lidar, y_lidar, z_lidar, intensity, side_range=(-50,50), fwd_range=(-50, 50), res= 0.2, min_height=-2.73, max_height=1.27)
if __name__ == '__main__':
    main()