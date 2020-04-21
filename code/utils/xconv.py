import tensorflow as tf
import pointfly as pf
import tf_util
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'grouping'))
sys.path.append(os.path.join(ROOT_DIR, '3d_interpolation'))
from tf_grouping import query_ball_point, group_point
from tf_interpolate import three_nn, three_interpolate

def channel_spatial_attention(input, is_training, bn_decay, bn=True, name=""):
    # input: [B, N, C]
    input = tf.expand_dims(input, axis=2)  # [B, N, 1, C]

    with tf.variable_scope("attention_"+name):
        feature_dim = input.get_shape()[-1].value

        maxpool_channel = tf.reduce_max(tf.reduce_max(input, axis=1, keep_dims=True), axis=2, keep_dims=True)  # [B, 1, 1, C]
        avgpool_channel = tf.reduce_mean(tf.reduce_mean(input, axis=1, keep_dims=True), axis=2, keep_dims=True) # [B, 1, 1, C]

        # shared MLP
        with tf.variable_scope("share_mlp"):
            mlp_1_max = tf_util.conv2d(maxpool_channel, feature_dim, [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=bn, is_training=is_training,
                                          scope='mlp_1', bn_decay=bn_decay) #, activation_fn=None)
            mlp_2_max = tf_util.conv2d(mlp_1_max, feature_dim, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=bn, is_training=is_training,
                                       scope='mlp_2', bn_decay=bn_decay) #, activation_fn=None)
        with tf.variable_scope("share_mlp", reuse=True):
            mlp_1_avg = tf_util.conv2d(avgpool_channel, feature_dim, [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=bn, is_training=is_training,
                                          scope='mlp_1', bn_decay=bn_decay) #, activation_fn=None)
            mlp_2_avg = tf_util.conv2d(mlp_1_avg, feature_dim, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=bn, is_training=is_training,
                                       scope='mlp_2', bn_decay=bn_decay) #, activation_fn=None)

        channel_atten = tf.nn.sigmoid(mlp_2_max+mlp_2_avg)  # [B, 1, 1, C]

        channel_refined_output = input * channel_atten  # [B, N, 1, C]

        maxpool_spatial = tf.reduce_max(input, axis=3, keep_dims=True)  # [B, N, 1, 1]
        avgpool_spatial = tf.reduce_mean(input, axis=3, keep_dims=True)  # [B, N, 1, 1]
        max_avg_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)  # [B, N, 1, 2]

        conv_layer = tf_util.conv2d(max_avg_spatial, 1, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=bn, is_training=is_training,
                                   scope='mlp_spatial', bn_decay=bn_decay) #, activation_fn=None)  # [B, N, 1, 1]
        spatial_attention = tf.nn.sigmoid(conv_layer)
        # conv_layer = tf.squeeze(conv_layer)  # [B, N]
        # pt_num = conv_layer.get_shape()[-1].value
        # min_val = tf.reduce_min(conv_layer, axis=-1, keepdims=True)  # [B, 1]
        # min_val = tf.tile(min_val, [1, pt_num])
        # max_val = tf.reduce_max(conv_layer, axis=-1, keepdims=True)  # [B, 1]
        # max_val = tf.tile(max_val, [1, pt_num])
        # spatial_attention = (conv_layer-min_val)/(max_val-min_val)
        # spatial_attention = tf.reshape(spatial_attention, shape=(-1, pt_num, 1, 1))

        output = channel_refined_output*spatial_attention  # [B, N, 1, C]

        output = tf.squeeze(output, axis=2)  # [B, N, C]

        return output, tf.squeeze(spatial_attention)

def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
    """
    pts: points of previous layer,
    fts: point features of previous layer,
    qrs: selected representative points of this layer,
    N: batch_size,
    K: neighbor_size,
    D: dilation parameter,
    P: the number of selected representative points,
    C: output feature number per point,
    C_pts_fts: feature number for each local point
    """
    if D == 1:
        _, indices = pf.knn_indices_general(qrs, pts, K, True)
    else:
        _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
        indices = indices_dilated[:, :, ::D, :]

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training, with_bn=False)  # following paper
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training, with_bn=False)  # following paper
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    if with_X_transformation:
        ######################## X-transformation #########################
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K), with_bn=False)  # following paper
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K), with_bn=False)  # following paper
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), with_bn=False, activation=None)  # following paper
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier, with_bn=True)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training, with_bn=True)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global_', is_training, with_bn=True)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d


def xconv_new(pts, fts, qrs, tag, N, K, radius, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
            D=1, sorting_method=None, with_global=False, knn=False):
    """
    pts: points of previous layer,
    fts: point features of previous layer,
    qrs: selected representative points of this layer,
    N: batch_size,
    K: neighbor_size,
    D: dilation parameter,
    P: the number of selected representative points,
    C: output feature number per point,
    C_pts_fts: feature number for each local point
    radius: float32, the radius of query ball search
    knn: True: knn; False: query ball
    """
    if knn:
        _, indices_dilated = pf.knn_indices_general(qrs, pts, K*D, True)
        indices = indices_dilated[:, :, ::D, :]
        nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
        if fts is not None:
            nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
    else:
        indices, pts_cnt = query_ball_point(radius, K, pts, qrs)
        nn_pts = group_point(pts, indices)
        if fts is not None:
            nn_fts_from_prev = group_point(fts, indices)

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)


    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    if with_X_transformation:
        ######################## X-transformation #########################
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))  # following paper
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))  # following paper
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)  # following paper
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global_', is_training)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d

def deconv(pts, fts, qrs, qrs_fts, C, tag, is_training):
    """
    pts: points of previous layer, e.g.,(B, N/2, 3)
    fts: point features of previous layer, e.g.,(B, N/2, C)
    qrs: selected representative points of this layer, e.g.,(B, N, 3)
    qrs_fts: e.g., (B, N, C)
    """
    dist, idx = three_nn(qrs, pts)
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
    norm = tf.tile(norm, [1, 1, 3])
    weight = (1.0 / dist) / norm
    interpolated_fts = three_interpolate(fts, idx, weight)  # (B, N, C)
    if qrs_fts is not None:
        interpolated_fts = tf.concat([interpolated_fts, qrs_fts], axis=2)

    interpolated_fts = tf.expand_dims(interpolated_fts, 2)
    new_features = pf.conv2d(interpolated_fts, C, tag, is_training=is_training, kernel_size=[1, 1])
    new_features = tf.squeeze(new_features, [2])

    return new_features


def deconv_new(pts, fts, qrs, tag, N, K, radius, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
            D=1, sorting_method=None, with_global=False, knn=False):
    """
    pts: points of previous layer, e.g.,(B, N/2, 3)
    fts: point features of previous layer, e.g.,(B, N/2, C)
    qrs: selected representative points of this layer, e.g.,(B, N, 3)
    N: batch_size,
    K: neighbor_size,
    D: dilation parameter,
    P: the number of selected representative points,
    C: output feature number per point,
    C_pts_fts: feature number for each local point
    radius: float32, the radius of query ball search
    knn: True: knn; False: query ball
    """
    dist, idx = three_nn(qrs, pts)
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
    norm = tf.tile(norm, [1, 1, 3])
    weight = (1.0 / dist) / norm
    interpolated_fts = three_interpolate(fts, idx, weight)  # (B, N, C)

    if knn:
        _, indices_dilated = pf.knn_indices_general(qrs, qrs, K*D, True)
        indices = indices_dilated[:, :, ::D, :]
        nn_pts = tf.gather_nd(qrs, indices, name=tag + 'nn_pts')  # (B, N, K, 3)
        nn_fts_from_prev = tf.gather_nd(interpolated_fts, indices, name=tag + 'nn_fts')
    else:
        indices, pts_cnt = query_ball_point(radius, K, qrs, qrs)
        nn_pts = group_point(qrs, indices)
        nn_fts_from_prev = group_point(interpolated_fts, indices)

    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (B, N, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (B, N, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)

    nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    if with_X_transformation:
        ######################## X-transformation #########################
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))  # following paper
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))  # following paper
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)  # following paper
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global_', is_training)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d


def sa_conv(pts, fts, qrs, K, D, mlp, bn, is_training, bn_decay, scope):
    """
    pts: points of previous layer,
    fts: point features of previous layer,
    qrs: selected representative points of this layer,
    N: batch_size,
    K: neighbor_size,
    D: dilation parameter,
    P: the number of selected representative points,
    C: output feature number per point,
    C_pts_fts: feature number for each local point
    """
    if D == 1:
        _, indices = pf.knn_indices_general(qrs, pts, K, True)
    else:
        _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
        indices = indices_dilated[:, :, ::D, :]

    new_points = tf.gather_nd(fts, indices)  # (N, P, K, 3)
    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
        new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        fts_conv_3d = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp[-1])

        return fts_conv_3d



def xconv_cov(cov, pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
    """
    cov: point covariance (B, N, 9) of previous layer
    pts: points of previous layer,
    fts: point features of previous layer,
    qrs: selected representative points of this layer,
    N: batch_size,
    K: neighbor_size,
    D: dilation parameter,
    P: the number of selected representative points,
    C: output feature number per point,
    C_pts_fts: feature number for each local point
    """
    if D == 1:
        _, indices = pf.knn_indices_general(qrs, pts, K, True)
    else:
        _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
        indices = indices_dilated[:, :, ::D, :]

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)

    nn_covs = tf.gather_nd(cov, indices, name=tag + 'nn_pts')  # (N, P, K, 9)
    # nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    # nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_covs, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training, with_bn=False)  # following paper
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training, with_bn=False)  # following paper
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    if with_X_transformation:
        ######################## X-transformation #########################
        X_0 = pf.conv2d(nn_covs, K * K, tag + 'X_0', is_training, (1, K), with_bn=False)  # following paper
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K), with_bn=False)  # following paper
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), with_bn=False, activation=None)  # following paper
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier, with_bn=True)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training, with_bn=True)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global_', is_training, with_bn=True)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d

