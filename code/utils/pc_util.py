""" Utility functions for processing point clouds.

Author: Charles R. Qi, Hao Su
Date: November 2016
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Draw point cloud
from eulerangles import euler2mat

# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement


# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


# a = np.zeros((16,1024,3))
# print point_cloud_to_volume_batch(a, 12, 1.0, False).shape

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


def point_cloud_to_volume_v2_batch(point_clouds, vsize=12, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxVxVxVxnum_samplex3
        Added on Feb 19
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume_v2(point_clouds[b, :, :], vsize, radius, num_sample)
        vol_list.append(np.expand_dims(vol, 0))
    return np.concatenate(vol_list, 0)


def point_cloud_to_volume_v2(points, vsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is vsize*vsize*vsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each voxel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    vol = np.zeros((vsize, vsize, vsize, num_sample, 3))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n, :])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n, :])
    # print loc2pc

    for i in range(vsize):
        for j in range(vsize):
            for k in range(vsize):
                if (i, j, k) not in loc2pc:
                    vol[i, j, k, :, :] = np.zeros((num_sample, 3))
                else:
                    pc = loc2pc[(i, j, k)]  # a list of (3,) arrays
                    pc = np.vstack(pc)  # kx3
                    # Sample/pad to num_sample points
                    if pc.shape[0] > num_sample:
                        choices = np.random.choice(pc.shape[0], num_sample, replace=False)
                        pc = pc[choices, :]
                    elif pc.shape[0] < num_sample:
                        pc = np.lib.pad(pc, ((0, num_sample - pc.shape[0]), (0, 0)), 'edge')
                    # Normalize
                    pc_center = (np.array([i, j, k]) + 0.5) * voxel - radius
                    # print 'pc center: ', pc_center
                    pc = (pc - pc_center) / voxel  # shift and scale
                    vol[i, j, k, :, :] = pc
                    # print (i,j,k), vol[i,j,k,:,:]
    return vol


def point_cloud_to_image_batch(point_clouds, imgsize, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxIxIxnum_samplex3
        Added on Feb 19
    """
    img_list = []
    for b in range(point_clouds.shape[0]):
        img = point_cloud_to_image(point_clouds[b, :, :], imgsize, radius, num_sample)
        img_list.append(np.expand_dims(img, 0))
    return np.concatenate(img_list, 0)


def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    img = np.zeros((imgsize, imgsize, num_sample, 3))
    pixel = 2 * radius / float(imgsize)
    locations = (points[:, 0:2] + radius) / pixel  # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n, :])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n, :])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i, j) not in loc2pc:
                img[i, j, :, :] = np.zeros((num_sample, 3))
            else:
                pc = loc2pc[(i, j)]
                pc = np.vstack(pc)
                if pc.shape[0] > num_sample:
                    choices = np.random.choice(pc.shape[0], num_sample, replace=False)
                    pc = pc[choices, :]
                elif pc.shape[0] < num_sample:
                    pc = np.lib.pad(pc, ((0, num_sample - pc.shape[0]), (0, 0)), 'edge')
                pc_center = (np.array([i, j]) + 0.5) * pixel - radius
                pc[:, 0:2] = (pc[:, 0:2] - pc_center) / pixel
                img[i, j, :, :] = pc
    return img


# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def draw_point_cloud(input_points, quality, canvasSize=500, space=200, diameter=3, add_diameter=5,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 1, 2], normalize=True, color=True, percentage=1.0):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
            quality: (N,) numpy array (layer output)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.ones((canvasSize, canvasSize, 3))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Order points by z-buffer
    # zorder = np.argsort(points[:, 2])
    # points = points[zorder, :]
    # points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    # max_depth = np.max(points[:, 2])
    #
    # ### change quality to RGB value
    # quality = quality[zorder]

    if np.max(quality[:]) > np.min(quality[:]):
        quality[:] = (quality[:] - np.min(quality[:])) / (np.max(quality[:] - np.min(quality[:])))
    else:
        color = False  # all the point qualities are equal, no need to colorize
    # if np.max(quality) == np.min(quality):
    #     color = False

    # if np.max(quality) != 0:
    #     quality = quality / np.max(quality)
    # else:
    #     color = False  # all the point qualities are equal, no need to colorize
    intensity = np.zeros((input_points.shape[0], 3))
    leftNum = int((quality.shape[0]) * percentage)
    quality_sorted = sorted(quality, reverse=True)
    thresholdVal = quality_sorted[leftNum-1]
    for i in range(quality.shape[0]):
        if color == True:  # render point color according to quality value
            if quality[i] < thresholdVal:
                quality[i] = 2.0  # black color
            if quality[i] >= thresholdVal and thresholdVal < 1.0:
                quality[i] = ((quality[i] - thresholdVal) / (1.0 - thresholdVal))
            if quality[i] >= thresholdVal and thresholdVal == 1.0:
                quality[i] = 1.0
        else:
            quality[i] = 2.0  # do not render point color


    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    ## render important points with larger radius
    diameter_large = diameter + add_diameter
    radius_large = (diameter_large - 1) / 2.0
    disk_large = np.zeros((diameter_large, diameter_large))
    for i in range(diameter_large):
        for j in range(diameter_large):
            if (i - radius_large) * (i - radius_large) + (j - radius_large) * (
                j - radius_large) <= radius_large * radius_large:
                disk_large[i, j] = np.exp((-(i - radius_large) ** 2 - (j - radius_large) ** 2) / (radius_large ** 2))
    mask_large = np.argwhere(disk_large > 0)
    dx_large = mask_large[:, 0]
    dy_large = mask_large[:, 1]
    dv_large = disk_large[disk_large > 0]

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        # transferred value = [(y2-y1)/(x2-x1)]*x+(x2*y1-x1*y2)/(x2-x1)
        if quality[j] < 0.125:
            intensity[j, 2] = ((1.0 - 0.5) / (0.125 - 0.0)) * quality[j] + (0.125 * 0.5 - 0.0 * 1.0) / (0.125 - 0.0)
        elif quality[j] < 0.375 and quality[j] >= 0.125:
            intensity[j, 2] = 1.0
            intensity[j, 1] = ((1.0 - 0.0) / (0.375 - 0.125)) * quality[j] + (0.375 * 0.0 - 0.125 * 1.0) / (0.375 - 0.125)
        elif quality[j] >= 0.375 and quality[j] < 0.625:
            intensity[j, 2] = ((0.0 - 1.0) / (0.625 - 0.375)) * quality[j] + (0.625 * 1.0 - 0.375 * 0.0) / (0.625 - 0.375)
            intensity[j, 1] = 1.0
            intensity[j, 0] = ((1.0 - 0.0) / (0.625 - 0.375)) * quality[j] + (0.625 * 0.0 - 0.375 * 1.0) / (0.625 - 0.375)
        elif quality[j] >= 0.625 and quality[j] < 0.875:
            intensity[j, 1] = ((0.0 - 1.0) / (0.875 - 0.625)) * quality[j] + (0.875 * 1.0 - 0.625 * 0.0) / (0.875 - 0.625)
            intensity[j, 0] = 1.0
        elif quality[j] >= 0.875 and quality[j] <= 1.0:
            intensity[j, 0] = ((0.5 - 1.0) / (1.0 - 0.875)) * quality[j] + (1.0 * 1.0 - 0.875 * 0.5) / (1.0 - 0.875)

        if quality[j] >= 0.625 and quality[j] <= 1.0:
            px = dx_large + xc
            py = dy_large + yc
            dv_large[:] = intensity[j, 0]
            image[px, py, 0] = dv_large
            dv_large[:] = intensity[j, 1]
            image[px, py, 1] = dv_large
            dv_large[:] = intensity[j, 2]
            image[px, py, 2] = dv_large
        else:
            px = dx + xc
            py = dy + yc
            dv[:] = intensity[j, 0]
            image[px, py, 0] = dv
            dv[:] = intensity[j, 1]
            image[px, py, 1] = dv
            dv[:] = intensity[j, 2]
            image[px, py, 2] = dv

    # image = image / np.max(image)
    # val = np.max(image)
    # val = np.percentile(image, 99.9)
    # image = image / val
    # mask = image==0
    # image[image>1.0] = 1.0
    # image = 1.0-image
    # image[mask] = 1.0

    return image


def point_cloud_three_views(points, quality):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, quality, zrot=0 / 180.0 * np.pi, xrot=0 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img2 = draw_point_cloud(points, quality, zrot=0 / 180.0 * np.pi, xrot=0 / 180.0 * np.pi, yrot=90 / 180.0 * np.pi)
    img3 = draw_point_cloud(points, quality, zrot=90.0 / 180.0 * np.pi, xrot=0 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


def point_cloud_three_views_demo():
    """ Demo for draw_point_cloud function """
    from PIL import Image
    points = read_ply('../third_party/mesh_sampling/piano.ply')
    im_array = point_cloud_three_views(points)
    img = Image.fromarray(np.uint8(im_array * 255.0))
    img.save('piano.jpg')


if __name__ == "__main__":
    point_cloud_three_views_demo()


def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)


def write_ply_color(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    import matplotlib.pyplot as pyplot
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x * 255) for x in c]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()
