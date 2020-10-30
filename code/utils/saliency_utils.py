import os
import sys
from tqdm import tqdm
import copy
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pygraph.algorithms
import pygraph.algorithms.minmax
import pygraph.classes.graph
from scipy import spatial
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util


def draw_activation(input_pts, input_qualities, log_dir, folder, method='region'):
    """input_pts: (M, num_point, 3)
    input_qualities: (M, num_point, 1)
    state: eval or test
    epoch: the iteration of training
    method: region or point
    """
    model_num = input_pts.shape[0]
    total_path = ROOT_DIR + "/" + log_dir + "/" + folder + "/html/"
    print total_path
    if not os.path.exists(total_path):
        os.makedirs(total_path)

    for i in range(model_num):
        input_gt = input_pts[i, :, :]
        quality = input_qualities[i, :, 0]

        quality1 = copy.deepcopy(quality)
        quality2 = copy.deepcopy(quality)
        quality3 = copy.deepcopy(quality)
        if method == 'point':
            diameter = 5
        else:
            diameter = 5

        img1 = pc_util.draw_point_cloud(input_gt, quality1, add_diameter=diameter,
                                        zrot=110 / 180.0 * np.pi,
                                        xrot=45 / 180.0 * np.pi,
                                        yrot=0 / 180.0 * np.pi,
                                        normalize=False, color=True)
        img2 = pc_util.draw_point_cloud(input_gt, quality2, add_diameter=diameter,
                                        zrot=70 / 180.0 * np.pi,
                                        xrot=135 / 180.0 * np.pi,
                                        yrot=90 / 180.0 * np.pi,
                                        normalize=False, color=True)
        img3 = pc_util.draw_point_cloud(input_gt, quality3, add_diameter=diameter,
                                        zrot=180.0 / 180.0 * np.pi,
                                        xrot=90 / 180.0 * np.pi,
                                        yrot=0 / 180.0 * np.pi,
                                        normalize=False, color=True)
        im_array = np.concatenate([img1, img2, img3], 1)
        img = Image.fromarray(np.uint8(im_array * 255.0), 'RGB')
        img_path = total_path + str(i) + ".jpg"
        img.save(img_path)

    ##################################################################
    html_name = "vis_" + method + ".html"
    index_path = os.path.join(total_path, html_name)
    index = open(index_path, "w")
    index.write("<html><body><table><tr>")

    # get sample list
    items = os.listdir(total_path)
    items.sort()

    # write img to file
    for item in items:
        if item.endswith(".jpg"):
            index.write("<tr>")
            index.write("<td>%s</td>" % item)
            id = item.split(".")[0]
            img_path = "%s.jpg" % (id)
            index.write("<td><img width='100%%', src='%s'></td>" % img_path)
    index.close()
    ##################################################################

def draw_activation_layer(input_pts, input_qualities, log_dir, layer='layer1'):
    """input_pts: (num_point, 3)
    input_qualities: (num_point, C)
    """
    total_path = ROOT_DIR + "/" + log_dir + "/" + layer + "/"
    if not os.path.exists(total_path):
        os.makedirs(total_path)

    feature_diml = input_qualities.shape[-1]
    for i in range(feature_diml):
        quality = input_qualities[:, i]

        quality1 = copy.deepcopy(quality)
        quality2 = copy.deepcopy(quality)
        quality3 = copy.deepcopy(quality)
        diameter = 5

        img1 = pc_util.draw_point_cloud(input_pts, quality1,
                                        zrot=110 / 180.0 * np.pi,
                                        xrot=45 / 180.0 * np.pi,
                                        yrot=0 / 180.0 * np.pi,
                                        color=True, percentage=1, add_diameter=diameter)
        img2 = pc_util.draw_point_cloud(input_pts, quality2,
                                        zrot=70 / 180.0 * np.pi,
                                        xrot=135 / 180.0 * np.pi,
                                        yrot=90 / 180.0 * np.pi,
                                        color=True, percentage=1, add_diameter=diameter)
        img3 = pc_util.draw_point_cloud(input_pts, quality3,
                                        zrot=180.0 / 180.0 * np.pi,
                                        xrot=90 / 180.0 * np.pi,
                                        yrot=0 / 180.0 * np.pi,
                                        color=True, percentage=1, add_diameter=diameter)
        im_array = np.concatenate([img1, img2, img3], 1)
        img = Image.fromarray(np.uint8(im_array * 255.0), 'RGB')
        img_path = total_path + str(i) + ".jpg"
        img.save(img_path)

    ##################################################################
    html_name = "vis_" + layer + ".html"
    index_path = os.path.join(total_path, html_name)
    index = open(index_path, "w")
    index.write("<html><body><table><tr>")

    # get sample list
    items = os.listdir(total_path)
    items.sort()

    # write img to file
    for item in items:
        if item.endswith(".jpg"):
            index.write("<tr>")
            index.write("<td>%s</td>" % item)
            id = item.split(".")[0]
            img_path = "%s.jpg" % (id)
            index.write("<td><img width='100%%', src='%s'></td>" % img_path)
    index.close()
    ##################################################################

def saliency_smooth(raw_data, prob):
    """
    :param raw_data:  (pts_num, 3)
    :param prob: (pts_num)
    :return: prob_smooth: (pts_num)
    """
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(raw_data)
    _, indices = nbrs.kneighbors(raw_data)
    pts_num = raw_data.shape[0]
    prob_smooth = np.zeros(shape=(pts_num))
    for i in range(pts_num):
        neighbor_quality = prob[indices[i]]
        prob_smooth[i] = np.mean(neighbor_quality)

    return prob_smooth

def compute_influence_distance(raw_data, prob):
    pts_num = raw_data.shape[0]
    graph = build_knn_graph(raw_data)
    delta = np.zeros(shape=(pts_num))

    ## process non_maximum points
    prob_max = np.max(prob)
    for i in range(pts_num):
        _, dist = pygraph.algorithms.minmax.shortest_path(graph, i)
        dist_list = np.asarray([dist[item] if dist.has_key(item) else 10000 for item in xrange(pts_num)])
        if prob[i] == prob_max:
            delta[i] = np.max(dist_list)
        else:
            indices = np.where(prob>prob[i])[0]
            delta[i] = np.min(dist_list[indices])

    # ## process maximum points
    # delta_max = np.max(delta)
    # for i in range(pts_num):
    #     if prob[i] == prob_max:
    #         delta[i] = delta_max

    ## normalize
    delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))

    return delta

def lambda_estimator_old(k, ordered_list):
    ## Hill estimator
    total_sum = 0.0
    for i in range(k):
        total_sum = total_sum + np.log(ordered_list[i]/ordered_list[k])

    return np.float(k+1)/total_sum

def lambda_estimator(m, gamma):
    pts_num = gamma.shape[0]
    k = int(math.ceil(0.95 * pts_num))

    ln_sum = 0
    for i in range(m+1, k+1):
        ln_sum = ln_sum + np.log(gamma[i]+0.0000001)

    first_term = (m / np.float(k-m+1)) * np.log(gamma[m+1]+0.0000001)
    second_term = (k / np.float(k-m+1)) * np.log(gamma[k+1]+0.0000001)
    third_term = (1.0 / np.float(k-m+1)) * ln_sum
    lambda_value = 1.0 / (first_term - second_term + third_term)

    return lambda_value

def select_key_points(gamma):
    pts_num = gamma.shape[0]
    m = int(math.ceil(0.1*pts_num))

    indices_sorted = np.argsort(-1*gamma)
    values_sorted = -np.sort(-gamma)
    # np.savetxt('/data/xzli/PointCloudSaliency_new/code/prob.txt', values_sorted, fmt='%.6f')
    # lambda_value = lambda_estimator(m, values_sorted)

    key_pts_label = np.zeros(shape=(pts_num))
    # threshold_idx = 0
    # base = 1.0 - math.pow(0.95, 1.0/np.float(m))
    # for k in range(m, -1, -1): # k+1 largest points since index starts from 0
    #     if k < 2:
    #         threshold_idx = k
    #         break
    #     lambda_value = lambda_estimator_old(k, values_sorted)
    #     index = -1.0/(np.float(k)*lambda_value)
    #     critical_value = math.pow(base, index)
    #     if (values_sorted[k]/values_sorted[k+1]) >= critical_value:
    #         threshold_idx = k
    #         break

    threshold_idx = 2
    key_pts_label[indices_sorted[0:threshold_idx]] = 1

    return key_pts_label

def select_key_points_new(gamma):
    pts_num = gamma.shape[0]

    indices_sorted = np.argsort(-1*gamma)
    values_sorted = -np.sort(-gamma)
    # np.savetxt('/data/xzli/PointCloudSaliency_new/code/prob.txt', values_sorted, fmt='%.6f')
    # lambda_value = lambda_estimator(m, values_sorted)

    pts_slope = np.zeros(shape=(pts_num))
    min_num = 8
    for i in range(min_num, pts_num-min_num):
        pts_slope[i] = (values_sorted[i-min_num] - values_sorted[i]) - (values_sorted[i] - values_sorted[i + min_num])

    # slope_indices_sorted = np.argsort(-1*pts_slope)
    # slope_sorted = -np.sort(-pts_slope)
    # pts_slope_slope = np.zeros(shape=(pts_num))
    # for i in range(1, pts_num - 1):
    #     pts_slope_slope[i] = slope_sorted[i - 1] - slope_sorted[i + 1]

    # threshold_idx = np.argmax(pts_slope_slope)
    # threshold_idx = slope_indices_sorted[threshold_idx]

    threshold_idx = np.argmax(pts_slope)
    key_pts_label = np.zeros(shape=(pts_num))
    key_pts_label[indices_sorted[0:(threshold_idx+1)]] = 1

    return key_pts_label

def decision_graph(data, qualities):
    """
    :param data:  [batch_size, num_point, 3]
    :param qualities: [batch_size, num_point, 1]
    :return: qualities_processed: [batch_size, num_point, 1]
    """
    model_num = data.shape[0]
    pts_num = data.shape[1]
    qualities_processed = np.zeros(shape=(model_num, pts_num, 1))
    for model_idx in tqdm(range(model_num)):
        raw_data = data[model_idx, :, :]  # (pts_num, 3)
        prob = qualities[model_idx, :, 0]  # (pts_num)

        if np.max(prob) != 0:
            prob = prob / np.max(prob)

        ## first step: saliency probability smoothing
        prob_smooth = prob
        # prob_smooth = saliency_smooth(raw_data, prob)

        ## second step: compute influence distance
        delta = compute_influence_distance(raw_data, prob_smooth)

        ## third step: compute the final value
        # indices_sorted = np.argsort(-1 * prob_smooth)
        # keep_num = int(pts_num*0.05)
        # keep_indices = indices_sorted[0:keep_num]
        # gamma = np.zeros(shape=(pts_num))
        # gamma[keep_indices] = prob_smooth[keep_indices]*delta[keep_indices]
        gamma = prob_smooth * delta

        ## select key points via 'outward statistical testing method'
        key_pts_label = select_key_points(gamma)
        # key_pts_label = select_key_points_new(gamma)

        qualities_processed[model_idx, :, 0] = key_pts_label

    return qualities_processed

def build_knn_graph(raw_data):
    """
    :param raw_data: (pts_num, 3)
    :return:
    """
    nbrs = spatial.cKDTree(raw_data)
    dists, idxs = nbrs.query(raw_data, k=5)

    ## build graph
    graph = pygraph.classes.graph.graph()
    graph.add_nodes(xrange(len(raw_data)))
    sid = 0
    for idx, dist in zip(idxs, dists):
        for eid, d in zip(idx, dist):
            if not graph.has_edge((sid, eid)) and eid < len(raw_data):
                graph.add_edge((sid, eid), d)
        sid = sid + 1

    return graph

def geodesic_knn(graph, raw_data, seed, r):
    """
    :param graph: knn graph
    :param raw_data: (pts_num, 3)
    :param seed: int, calculate the distance from this seed point to all the other points
    :param r: neighbor radius
    :return:
    """
    pts_num = raw_data.shape[0]

    ## dijkstra algorithm
    _, dist = pygraph.algorithms.minmax.shortest_path(graph, seed)
    dist_list = np.asarray([dist[item] if dist.has_key(item) else 10000 for item in xrange(pts_num)])
    neighbor_idx = np.where(dist_list<=r)[0]

    return neighbor_idx

def metrics_calculation(qualities_processed, schelling_data, r=0.04):
    """
    :param qualities_processed: [model_num, num_point, 1]
    :param test_data: [model_num, num_point, 5]  4th column is the ground truth point label; 5th column is the gt prob
    :return: metrics: [model_num, 3]
    """
    model_num = qualities_processed.shape[0]
    metrics = np.zeros(shape=(model_num, 3))
    for model_idx in range(model_num):
        raw_data = schelling_data[model_idx, :, 0:3]
        label = qualities_processed[model_idx, :, 0]
        # extract ground truth points
        gt_indices = np.where(schelling_data[model_idx, :, 3]==1)[0]
        gt_pts = schelling_data[model_idx, gt_indices, 0:3]
        gt_pts_prob = schelling_data[model_idx, gt_indices, 4]

        # construct knn graph
        graph = build_knn_graph(raw_data)

        # ============FNE metric calculation=============
        gt_pts_num = gt_pts.shape[0]
        detected_label = np.zeros(shape=(gt_pts_num))
        N_c = 0
        for i in range(gt_pts_num):
            seed = np.where((raw_data == gt_pts[i, :]).all(axis=1))[0]
            seed = set(seed).pop()
            neighbor_idx = geodesic_knn(graph, raw_data, seed, r)
            if 1.0 in label[neighbor_idx]:
                N_c = N_c + 1
                detected_label[i] = 1.0

        FNE = 1.0 - (np.float(N_c) / np.float(gt_pts_num))
        # ================================================

        # =============FPE metric calculation=============
        N_a = sum(label == 1)
        N_f = N_a - N_c
        FPE = np.float(N_f) / np.float(N_a)
        # ================================================

        # =============WME metric calculation=============
        value = 0.0
        for i in range(gt_pts_num):
            value = value + gt_pts_prob[i] * detected_label[i]
        WME = 1.0 - (np.float(value) / np.float(np.sum(gt_pts_prob)))
        # ================================================

        metrics[model_idx, 0] = FNE
        metrics[model_idx, 1] = FPE
        metrics[model_idx, 2] = WME

    # save the metrics into a file
    np.savetxt('metrics.txt', metrics, fmt='%.6f')
    return metrics